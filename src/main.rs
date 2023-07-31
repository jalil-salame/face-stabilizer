use std::ops::Deref;
use std::path::PathBuf;

use anyhow::anyhow;
use anyhow::bail;
use anyhow::Context;
use clap::Parser;
use clap::Subcommand;
use dlib_face_recognition::FaceDetector;
use dlib_face_recognition::ImageMatrix;
use dlib_face_recognition::LandmarkPredictor;
#[cfg(feature = "gui")]
use iced::Sandbox;
use landmark_extractor::Faces;
use log::debug;
use log::error;
use log::info;
use log::warn;

#[derive(Debug, Parser)]
struct Opts {
    #[command(subcommand)]
    command: Actions,
}

macro_rules! log_err_bail {
    ($e:expr) => {
        match $e {
            Ok(v) => v,
            Err(err) => {
                error!("{err}");
                return;
            }
        }
    };
}

#[derive(Debug, Subcommand)]
enum Actions {
    /// Extract Features from images to process later
    ExtractFeatures {
        /// Path to the Shape Predictor model (also called Facial Landmarks Predictor)
        #[arg(env, short, long)]
        shape_predictor: PathBuf,
        /// Path to a directory containing the images you want to extract the features of
        image_dir: PathBuf,
        /// Path to the output file
        #[arg(short, long, default_value = "landmarks.ron")]
        output: PathBuf,
        /// Whether to pretty print the extracted text
        #[arg(short, long)]
        pretty: bool,
    },
    /// Launch a GUI
    #[cfg(feature = "gui")]
    GUI,
}

fn main() -> anyhow::Result<()> {
    // Configure using RUST_LOG=* (ie. RUST_LOG=info)
    env_logger::init();

    let opts = Opts::parse();
    debug!("Recieved {opts:?}");

    match opts.command {
        Actions::ExtractFeatures {
            shape_predictor,
            image_dir,
            output,
            pretty,
        } => extract_features(shape_predictor, image_dir, output, pretty),
        #[cfg(feature = "gui")]
        Actions::GUI => Gui::run(iced::Settings {
            // default_font: iced::Font::with_name("DejaVu Sans"),
            ..iced::Settings::default()
        })
        .context("running gui"),
    }
}

#[cfg(feature = "gui")]
#[derive(Debug, Default, Clone)]
enum Message {
    #[default]
    NoOp,
    SelectFeaturesFile,
}

#[derive(Debug, Default, Clone)]
#[cfg(feature = "gui")]
struct Gui {
    images: Vec<PathBuf>,
    features: std::collections::HashMap<PathBuf, Faces>,
}

#[cfg(feature = "gui")]
impl iced::Sandbox for Gui {
    type Message = Message;

    fn new() -> Self {
        Self::default()
    }

    fn title(&self) -> String {
        "Extract Facial Features".to_string()
    }

    fn update(&mut self, message: Self::Message) {
        match message {
            Message::NoOp => {}
            Message::SelectFeaturesFile => {
                if let Some(file) = rfd::FileDialog::new()
                    .set_title("Open Encoded Features")
                    .pick_file()
                {
                    let data =
                        log_err_bail!(std::fs::read(&file)
                            .with_context(|| format!("reading {}", file.display())));
                    self.features =
                        log_err_bail!(ron::de::from_bytes(&data).context("decoding features"));
                }
            }
        }
    }

    fn view(&self) -> iced::Element<'_, Self::Message> {
        use iced::widget::button;
        use iced::widget::column;

        column![
            iced::widget::vertical_space(iced::Length::Fill),
            button("Open Encoded Features").on_press(Message::SelectFeaturesFile),
            iced::widget::vertical_space(iced::Length::Fill),
        ]
        .align_items(iced::Alignment::Center)
        .spacing(8)
        .into()
    }
}

fn extract_features(
    shape_predictor: PathBuf,
    image_dir: PathBuf,
    output: PathBuf,
    pretty: bool,
) -> anyhow::Result<()> {
    if output.exists() {
        warn!("{} exists, making a backup", output.display());
        let mut backup = output.clone();
        backup.set_extension(output.extension().map_or("bak".to_string(), |ext| {
            format!("{}.bak", ext.to_str().unwrap_or(""))
        }));
        std::fs::rename(&output, backup).context("trying to backup the ouput file")?;
    }
    let output = std::fs::File::create(output)?;

    let file = shape_predictor.display();
    info!("Loading shape predictor from {file}",);
    if !shape_predictor.is_file() {
        bail!("{file} is not a regular file (or doesn't exist).",);
    }
    let predictor = LandmarkPredictor::open(shape_predictor).map_err(|err| anyhow!(err))?;

    let image_paths: Vec<_> = std::fs::read_dir(image_dir)
        .context("trying to open image_dir")?
        .filter_map(|dir_ent| -> Option<anyhow::Result<PathBuf>> {
            let Ok(dir_ent) = dir_ent else { return Some(Err(dir_ent.unwrap_err().into())) };
            let ft = match dir_ent.file_type().with_context(|| {
                format!(
                    "trying to get the file type of {}",
                    dir_ent.file_name().to_string_lossy()
                )
            }) {
                Ok(ft) => ft,
                Err(err) => return Some(Err(err)),
            };
            if !ft.is_file() {
                info!(
                    "{} is not a file, skipping",
                    dir_ent.file_name().to_string_lossy()
                );
                return None;
            }
            Some(Ok(dir_ent.path()))
        })
        .collect::<anyhow::Result<_>>()?;

    use indicatif::*;
    let style =
        ProgressStyle::with_template("[{pos:>4}/{len:4}] {msg} {bar} [{per_sec} {eta_precise}]")
            .expect("valid template");

    #[cfg(feature = "rayon")]
    use rayon::prelude::*;
    #[cfg(feature = "rayon")]
    let iter = image_paths.into_par_iter();

    #[cfg(not(feature = "rayon"))]
    let iter = image_paths.into_iter();

    let features: std::collections::HashMap<_, _> = iter
        .progress_with_style(style)
        .map(
            |path| -> anyhow::Result<(PathBuf, landmark_extractor::Faces)> {
                let img = image::open(&path)
                    .with_context(|| format!("failed to open {}", path.display()))?
                    .into_rgb8();
                let mat = ImageMatrix::from_image(&img);
                let detector = FaceDetector::new(); // Detector shouldn't be sync https://github.com/ulagbulag/dlib-face-recognition/issues/25
                let landmarks = landmark_extractor::extract_landmarks(&mat, &detector, &predictor);
                Ok((path, landmarks))
            },
        )
        .collect::<anyhow::Result<_>>()?;

    info!("finished processing");
    info!("serializing to file");
    if pretty {
        ron::ser::to_writer_pretty(output, &features, ron::ser::PrettyConfig::default())
    } else {
        ron::ser::to_writer(output, &features)
    }
    .context("serializing landmarks to a file")?;
    Ok(())
}
