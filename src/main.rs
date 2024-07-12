use std::collections::HashMap;
use std::ffi::OsStr;
use std::path::Path;
use std::path::PathBuf;

use clap::Parser;
use clap::Subcommand;
use dlib_face_recognition::FaceDetector;
use dlib_face_recognition::ImageMatrix;
use dlib_face_recognition::LandmarkPredictor;
use imageproc::geometric_transformations::warp;
use imageproc::geometric_transformations::Interpolation;
#[cfg(feature = "rayon")]
use indicatif::ParallelProgressIterator;
#[cfg(not(feature = "rayon"))]
use indicatif::ProgressIterator;
use indicatif::ProgressStyle;
use landmark_extractor::Faces;
use landmark_extractor::Landmarks;
use log::debug;
use log::info;
use log::warn;
use miette::bail;
use miette::ensure;
use miette::Context;
use miette::IntoDiagnostic;
use miette::Result;
#[cfg(feature = "rayon")]
use rayon::prelude::*;
use serde::Deserialize;
use serde::Serialize;

#[cfg(feature = "gui")]
mod gui;

fn main() -> Result<()> {
    // Pretty panics
    miette::set_panic_hook();
    // Configure using RUST_LOG=* (ie. RUST_LOG=info)
    env_logger::init();
    // Parse opts
    let opts = Opts::parse();
    debug!("Received {opts:?}");
    opts.run()
}

#[derive(Debug, Parser)]
struct Opts {
    #[command(subcommand)]
    command: Actions,
}

impl Opts {
    fn run(self) -> Result<()> {
        match self.command {
            Actions::ExtractFeatures {
                shape_predictor,
                image_dir,
                output,
                pretty,
            } => extract_features(&shape_predictor, &image_dir, &output, pretty),
            Actions::Transform {
                features,
                output_dir,
            } => transform_images(&features, &output_dir),
            #[cfg(feature = "gui")]
            Actions::GUI => gui::Gui::run(iced::Settings {
                // default_font: iced::Font::with_name("DejaVu Sans"),
                ..iced::Settings::default()
            })
            .context("running gui"),
        }
    }
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
    Transform {
        /// Path to the extracted features
        features: PathBuf,
        /// Directory where to place the transformed images
        #[arg(short, long, default_value = "./out")]
        output_dir: PathBuf,
    },
    /// Launch a GUI
    #[cfg(feature = "gui")]
    GUI,
}

fn transform_images(features: &Path, output_dir: &Path) -> Result<()> {
    ensure!(features.exists(), "could not find {}", features.display());
    ensure!(features.is_file(), "{} is not a file", features.display());
    let file = std::fs::File::open(features)
        .into_diagnostic()
        .context("opening features file")?;
    let Features { basedir, features }: Features = ron::de::from_reader(file)
        .into_diagnostic()
        .context("deserializing features")?;
    let mut features: Vec<_> = features.into_iter().collect();
    features.sort_by_key(|(fname, _features)| fname.clone());
    if !output_dir.exists() {
        std::fs::create_dir(output_dir)
            .into_diagnostic()
            .with_context(|| format!("creating {} directory", output_dir.display()))?;
    } else {
        ensure!(
            output_dir.is_dir(),
            "{} is not a directory",
            output_dir.display()
        );
    }
    let out_path = |path: &str| output_dir.join(path);

    let (ref_name, ref_feat) = features.swap_remove(0);
    let ref_path = basedir.join(ref_name.as_ref());
    ensure!(
        ref_feat.len() == 1,
        "reference face should have exactly one face"
    );
    let (_, ref_feat) = ref_feat.iter().next().unwrap().clone().into();
    std::fs::copy(ref_path.as_path(), out_path(&ref_name))
        .into_diagnostic()
        .with_context(|| {
            format!(
                "failed to copy {} to {}",
                ref_path.display(),
                out_path(&ref_name).display()
            )
        })?;

    use indicatif::*;
    let style =
        ProgressStyle::with_template("[{pos:>4}/{len:4}] {msg} {bar} [{per_sec} {eta_precise}]")
            .expect("valid template");

    #[cfg(feature = "rayon")]
    use rayon::prelude::*;
    #[cfg(feature = "rayon")]
    let features = features.into_par_iter();
    #[cfg(not(feature = "rayon"))]
    let features = features.into_iter();

    features
        .progress_with_style(style)
        .map(|(img_name, img_feat)| {
            let img_path = basedir.join(img_name.as_ref());
            if img_feat.len() != 1 {
                warn!(
                    "{} does not have a single face, it has {} instead",
                    img_path.display(),
                    img_feat.len()
                );
                return Ok(());
            }

            let (_, img_feat) = img_feat.iter().next().unwrap().clone().into();
            let img = image::open(&img_path)
                .into_diagnostic()
                .with_context(|| format!("opening image {}", img_path.display()))?
                .into_rgb8();

            let out = out_path(img_name.as_ref());

            apply_projection(&ref_feat, &img_feat, &img, [0, 0, 0].into())
                .save(&out)
                .into_diagnostic()
                .with_context(|| format!("saving image to {}", out.display()))
        })
        .collect()
}

fn extract_features(
    shape_predictor: &Path,
    basedir: &Path,
    output: &Path,
    pretty: bool,
) -> Result<()> {
    if output.exists() {
        warn!("{} exists, making a backup", output.display());
        let mut backup = output.to_path_buf();
        backup.set_extension(output.extension().map_or("bak".to_string(), |ext| {
            format!("{}.bak", ext.to_str().unwrap_or(""))
        }));
        std::fs::rename(output, backup)
            .into_diagnostic()
            .context("trying to backup the output file")?;
    }
    let output = std::fs::File::create(output).into_diagnostic()?;

    let file = shape_predictor.display();
    info!("Loading shape predictor from {file}",);
    if !shape_predictor.is_file() {
        bail!("{file} is not a regular file (or doesn't exist).",);
    }
    let predictor = LandmarkPredictor::open(shape_predictor).map_err(|err| miette::miette!(err))?;

    let imgs: Vec<_> = std::fs::read_dir(basedir)
        .into_diagnostic()
        .context("trying to open image_dir")?
        .filter_map(|dir_ent| {
            let dir_ent = match dir_ent {
                Ok(dir_ent) => dir_ent,
                Err(err) => {
                    return Some(Err(err).into_diagnostic());
                }
            };
            let ft = match dir_ent.file_type().into_diagnostic().with_context(|| {
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
            Some(Ok(dir_ent.file_name().into_boxed_os_str()))
        })
        .collect::<Result<_>>()?;

    use indicatif::*;
    let style =
        ProgressStyle::with_template("[{pos:>4}/{len:4}] {msg} {bar} [{per_sec} {eta_precise}]")
            .expect("valid template");

    let features = extract(basedir, imgs, style, &predictor)?;

    info!("finished processing");
    info!("serializing to file");
    if pretty {
        ron::ser::to_writer_pretty(output, &features, ron::ser::PrettyConfig::default())
    } else {
        ron::ser::to_writer(output, &features)
    }
    .into_diagnostic()
    .context("serializing landmarks to a file")?;
    Ok(())
}

#[cfg(not(feature = "rayon"))]
fn extract(
    basedir: &Path,
    imgs: Vec<Box<OsStr>>,
    style: ProgressStyle,
    predictor: &LandmarkPredictor,
) -> Result<Features> {
    let detector = FaceDetector::new();
    let features = imgs
        .into_iter()
        .progress_with_style(style)
        .map(|name| {
            let path = basedir.join(name.as_ref());
            let img = image::open(&path)
                .into_diagnostic()
                .with_context(|| format!("failed to open {}", path.display()))?
                .into_rgb8();
            let image = ImageMatrix::from_image(&img);
            let landmarks = landmark_extractor::extract_landmarks(&image, &detector, predictor);
            Ok((name, landmarks))
        })
        .collect::<Result<_>>()?;
    Ok(Features {
        basedir: basedir.to_path_buf(),
        features,
    })
}

#[cfg(feature = "rayon")]
fn extract(
    basedir: &Path,
    imgs: Vec<Box<OsStr>>,
    style: ProgressStyle,
    predictor: &LandmarkPredictor,
) -> Result<Features> {
    thread_local! {
        static DETECTOR: FaceDetector = FaceDetector::new();
    };
    let features = imgs
        .into_par_iter()
        .progress_with_style(style)
        .map(|name| {
            let path = basedir.join(name.as_ref());
            let img = image::open(&path)
                .into_diagnostic()
                .with_context(|| format!("failed to open {}", path.display()))?
                .into_rgb8();
            let image = ImageMatrix::from_image(&img);
            let landmarks = DETECTOR.with(|detector| {
                landmark_extractor::extract_landmarks(&image, detector, predictor)
            });
            Ok((
                name.to_string_lossy().to_string().into_boxed_str(),
                landmarks,
            ))
        })
        .collect::<Result<_>>()?;
    Ok(Features {
        basedir: basedir.into(),
        features,
    })
}

#[derive(Debug, Serialize, Deserialize)]
struct Features {
    /// The directory with all the images
    basedir: Box<Path>,
    /// Mapping from Filename -> Faces
    features: HashMap<Box<str>, Faces>,
}

fn apply_projection(
    target: &Landmarks,
    points: &Landmarks,
    image: &image::RgbImage,
    default: image::Rgb<u8>,
) -> image::ImageBuffer<image::Rgb<u8>, Vec<u8>> {
    let mut target: Vec<_> = target
        .iter()
        .map(|&(x, y)| (x as f32, y as f32).into())
        .collect();
    let mut points: Vec<_> = points
        .iter()
        .map(|&(x, y)| (x as f32, y as f32).into())
        .collect();
    let projection = stabilizer::procrustes_superimposition(&mut target, &mut points)
        .expect("couldn't project points into target");
    warp(image, &projection, Interpolation::Bicubic, default)
}
