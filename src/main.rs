use std::collections::HashMap;
use std::path::Path;
use std::path::PathBuf;

use anyhow::anyhow;
use anyhow::bail;
use anyhow::ensure;
use anyhow::Context;
use clap::Parser;
use clap::Subcommand;
use dlib_face_recognition::FaceDetector;
use dlib_face_recognition::ImageMatrix;
use dlib_face_recognition::LandmarkPredictor;
use imageproc::geometric_transformations::warp;
use imageproc::geometric_transformations::Interpolation;
use landmark_extractor::Faces;
use landmark_extractor::Landmarks;
use log::debug;
use log::info;
use log::warn;

#[cfg(feature = "gui")]
mod gui;

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
        Actions::Transform {
            features,
            output_dir,
        } => {
            ensure!(features.exists(), "could not find {}", features.display());
            ensure!(features.is_file(), "{} is not a file", features.display());
            let file = std::fs::File::open(features).context("opening features file")?;
            let features: Features =
                ron::de::from_reader(file).context("deserializing features")?;
            let mut features: Vec<_> = features.into_iter().collect();
            features.sort_by_cached_key(|f| f.0.clone());
            if !output_dir.exists() {
                std::fs::create_dir(&output_dir)
                    .with_context(|| format!("creating {} directory", output_dir.display()))?;
            } else {
                ensure!(
                    output_dir.is_dir(),
                    "{} is not a directory",
                    output_dir.display()
                );
            }
            let out_path =
                |file: &Path| output_dir.join(file.file_name().expect("valid file name"));

            let (ref_path, ref_feat) = features.swap_remove(0);
            ensure!(
                ref_feat.len() == 1,
                "reference face should have exactly one face"
            );
            let (_, ref_feat) = ref_feat.iter().next().unwrap().clone().into();
            std::fs::copy(&ref_path, out_path(&ref_path))?;

            use indicatif::*;
            let style = ProgressStyle::with_template(
                "[{pos:>4}/{len:4}] {msg} {bar} [{per_sec} {eta_precise}]",
            )
            .expect("valid template");

            #[cfg(feature = "rayon")]
            use rayon::prelude::*;
            #[cfg(feature = "rayon")]
            let features = features.into_par_iter();
            #[cfg(not(feature = "rayon"))]
            let features = features.into_iter();

            features
                .progress_with_style(style)
                .map(|(img_path, img_feat)| {
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
                        .with_context(|| format!("opening image {}", img_path.display()))?
                        .into_rgb8();

                    let out = out_path(&img_path);

                    apply_projection(&ref_feat, &img_feat, &img)
                        .save(&out)
                        .with_context(|| format!("saving image to {}", out.display()))
                })
                .collect()
        }
        #[cfg(feature = "gui")]
        Actions::GUI => gui::Gui::run(iced::Settings {
            // default_font: iced::Font::with_name("DejaVu Sans"),
            ..iced::Settings::default()
        })
        .context("running gui"),
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

    let features: Features = iter
        .progress_with_style(style)
        .map(|path| -> anyhow::Result<(PathBuf, Faces)> {
            let img = image::open(&path)
                .with_context(|| format!("failed to open {}", path.display()))?
                .into_rgb8();
            let mat = ImageMatrix::from_image(&img);
            let detector = FaceDetector::new(); // Detector shouldn't be sync https://github.com/ulagbulag/dlib-face-recognition/issues/25
            let landmarks = landmark_extractor::extract_landmarks(&mat, &detector, &predictor);
            Ok((path, landmarks))
        })
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

type Features = HashMap<PathBuf, Faces>;

fn apply_projection(
    target: &Landmarks,
    points: &Landmarks,
    image: &image::RgbImage,
) -> image::ImageBuffer<image::Rgb<u8>, Vec<u8>> {
    let target = target.iter().map(|&(x, y)| (x as f32, y as f32).into());
    let points = points.iter().map(|&(x, y)| (x as f32, y as f32).into());
    let proj = stabilizer::procrustes_superimposition(target, points)
        .expect("neither points nor target are empty and they have the same length");
    warp(image, &proj, Interpolation::Bicubic, image::Rgb([0, 0, 0]))
}
