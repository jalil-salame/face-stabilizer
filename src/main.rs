use anyhow::anyhow;
use anyhow::bail;
use anyhow::Context;
use clap::Parser;
use clap::Subcommand;
use dlib_face_recognition::FaceDetector;
use dlib_face_recognition::ImageMatrix;
use dlib_face_recognition::LandmarkPredictor;
use log::debug;
use log::info;
use log::warn;

#[derive(Debug, Parser)]
struct Opts {
    #[command(subcommand)]
    command: Actions,
}

#[derive(Debug, Subcommand)]
enum Actions {
    /// Extract Features from images to process later
    ExtractFeatures {
        /// Path to the Shape Predictor model (also called Facial Landmarks Predictor)
        #[arg(env, short, long)]
        shape_predictor: std::path::PathBuf,
        /// Path to a directory containing the images you want to extract the features of
        image_dir: std::path::PathBuf,
        /// Path to the output file
        #[arg(short, long, default_value = "landmarks.ron")]
        output: std::path::PathBuf,
    },
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
        } => {
            if output.exists() {
                warn!("{} exists, making a backup", output.display());
                let mut backup = output.clone();
                backup.set_extension(output.extension().map_or("bak".to_string(), |ext| {
                    format!("{}.bak", ext.to_str().unwrap_or(""))
                }));
                std::fs::rename(&output, backup).context("trying to backup the ouput file")?;
            }
            let output = std::fs::File::create(output)?;

            let detector = FaceDetector::new();

            let file = shape_predictor.display();
            info!("Loading shape predictor from {file}",);
            if !shape_predictor.is_file() {
                bail!("{file} is not a regular file (or doesn't exist).",);
            }
            let predictor = LandmarkPredictor::open(shape_predictor).map_err(|err| anyhow!(err))?;

            let image_paths: Vec<_> = std::fs::read_dir(image_dir)
                .context("trying to open image_dir")?
                .filter_map(|dir_ent| -> Option<anyhow::Result<std::path::PathBuf>> {
                    let Ok(dir_ent) = dir_ent else { return Some(Err(dir_ent.unwrap_err().into())) };
                    let ft = match dir_ent
                        .file_type()
                        .with_context(|| format!("trying to get the file type of {}", dir_ent.file_name().to_string_lossy()))
                    {
                        Ok(ft) => ft,
                        Err(err) => return Some(Err(err)),
                    };
                    if !ft.is_file() {
                        info!("{} is not a file, skipping", dir_ent.file_name().to_string_lossy());
                        return None;
                    }
                    Some(Ok(dir_ent.path()))
                }).collect::<anyhow::Result<_>>()?;

            use indicatif::ProgressIterator;
            let style = indicatif::ProgressStyle::with_template(
                "[{pos:>4}/{len:4}] {msg} {bar} [{per_sec} {eta_precise}]",
            )
            .expect("valid template");
            let features: std::collections::HashMap<_, _> = image_paths
                .into_iter()
                .progress_with_style(style)
                .map(
                    |path| -> anyhow::Result<(std::path::PathBuf, landmark_extractor::Faces)> {
                        let img = image::open(&path)
                            .with_context(|| format!("failed to open {}", path.display()))?
                            .into_rgb8();
                        let mat = ImageMatrix::from_image(&img);
                        let landmarks =
                            landmark_extractor::extract_landmarks(&mat, &detector, &predictor);
                        Ok((path, landmarks))
                    },
                )
                .collect::<anyhow::Result<_>>()?;

            info!("finished processing");
            info!("serializing to file");
            ron::ser::to_writer_pretty(output, &features, ron::ser::PrettyConfig::default())
                .context("serializing landmarks to a file")?;
        }
    }

    Ok(())
}
