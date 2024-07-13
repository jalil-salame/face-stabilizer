use std::borrow::Cow;
use std::collections::BTreeMap;
use std::path::Path;
use std::path::PathBuf;

use clap::Parser;
use clap::Subcommand;
use dlib_face_recognition::ImageMatrix;
use dlib_face_recognition::LandmarkPredictor;
use imageproc::geometric_transformations::warp;
use imageproc::geometric_transformations::Interpolation;
#[cfg(feature = "rayon")]
use indicatif::ParallelProgressIterator;
use indicatif::ProgressIterator;
use indicatif::ProgressStyle;
use landmark_extractor::extract_landmarks_cnn;
use landmark_extractor::extract_landmarks_fast;
use landmark_extractor::set_cnn_path;
use landmark_extractor::Face;
use landmark_extractor::Faces;
use landmark_extractor::Landmarks;
use log::debug;
use log::info;
use log::warn;
use miette::bail;
use miette::ensure;
use miette::IntoDiagnostic;
use miette::Result;
use miette::WrapErr;
#[cfg(feature = "rayon")]
use rayon::prelude::*;
use serde::Deserialize;
use serde::Serialize;

fn main() -> Result<()> {
    // Pretty panics
    miette::set_panic_hook();
    // Configure using RUST_LOG=* (ie. RUST_LOG=info)
    env_logger::builder()
        .filter_level(log::LevelFilter::Info)
        .parse_default_env()
        .init();
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
            Actions::Stabilize {
                shape_predictor,
                image_dir,
                output_dir,
                face_detector,
                use_cnn_detector,
                output,
                pretty,
            } => {
                // Load CNN
                if use_cnn_detector {
                    ensure!(
                        face_detector.is_some(),
                        "requested cnn detector but didn't provide a path to the model"
                    );
                    let Some(path) = face_detector else {
                        unreachable!()
                    };
                    info!("loading CNN model");
                    set_cnn_path(&path).map_err(|err| miette::miette!("{err}"))?;
                }
                // Backup previous results
                if output.exists() {
                    warn!("{} exists, making a backup", output.display());
                    let mut backup = output.to_path_buf();
                    backup.set_extension(output.extension().map_or("bak".to_string(), |ext| {
                        format!("{}.bak", ext.to_str().unwrap_or(""))
                    }));
                    std::fs::rename(&output, backup)
                        .into_diagnostic()
                        .wrap_err("trying to backup the output file")?;
                }
                // Ensure we can create the output file before extracting features
                let writer = std::fs::File::create(output).into_diagnostic()?;
                // Load shape predictor
                let file = shape_predictor.display();
                info!("Loading shape predictor from {file}",);
                if !shape_predictor.is_file() {
                    bail!("{file} is not a regular file (or doesn't exist).",);
                }
                let predictor =
                    LandmarkPredictor::open(shape_predictor).map_err(|err| miette::miette!(err))?;
                let mut imgs = all_file_names_in_dir(&image_dir)?;
                imgs.sort_unstable();
                ensure!(
                    !imgs.is_empty(),
                    "no images found in {}",
                    image_dir.display()
                );
                let base = imgs.remove(0);
                let (_name, faces) =
                    extract_feature(&image_dir, &predictor, use_cnn_detector)(base.clone())?;
                let Face(_rect, landmarks) = match faces.as_ref() {
                    [] => bail!("ignoring {base}: no faces found"),
                    [face] => face,
                    _ => {
                        let n = faces.len();
                        bail!("ignoring {base}: found {n} faces, cannot choose which to stabilize");
                    }
                };
                let features: Result<_> =
                    maybe_parallel("stabilizing images", imgs, !use_cnn_detector, |name| {
                        match stabilize_image(
                            &image_dir,
                            &output_dir,
                            use_cnn_detector,
                            &predictor,
                            landmarks,
                        )(name)
                        {
                            Ok(data) => Some(Ok(data?)),
                            Err(err) => Some(Err(err)),
                        }
                    });
                let features = Features {
                    basedir: image_dir.into_boxed_path(),
                    features: features?,
                };
                // Serialize results
                info!("serializing to file");
                if pretty {
                    serde_json::ser::to_writer_pretty(writer, &features)
                } else {
                    serde_json::ser::to_writer(writer, &features)
                }
                .into_diagnostic()
                .wrap_err("serializing landmarks to a file")
            }
            Actions::ExtractFeatures {
                shape_predictor,
                image_dir,
                output,
                pretty,
                face_detector,
                use_cnn_detector,
            } => {
                if use_cnn_detector {
                    ensure!(
                        face_detector.is_some(),
                        "requested cnn detector but didn't provide a path to the model"
                    );
                    let Some(path) = face_detector else {
                        unreachable!()
                    };
                    set_cnn_path(&path).map_err(|err| miette::miette!("{err}"))?;
                }
                // Backup previous results
                if output.exists() {
                    warn!("{} exists, making a backup", output.display());
                    let mut backup = output.to_path_buf();
                    backup.set_extension(output.extension().map_or("bak".to_string(), |ext| {
                        format!("{}.bak", ext.to_str().unwrap_or(""))
                    }));
                    std::fs::rename(&output, backup)
                        .into_diagnostic()
                        .wrap_err("trying to backup the output file")?;
                }
                // Ensure we can create the output file before extracting features
                let writer = std::fs::File::create(output).into_diagnostic()?;
                // Extract features
                let features = extract_features(&shape_predictor, &image_dir, use_cnn_detector)?;
                // Serialize results
                info!("serializing to file");
                if pretty {
                    serde_json::ser::to_writer_pretty(writer, &features)
                } else {
                    serde_json::ser::to_writer(writer, &features)
                }
                .into_diagnostic()
                .wrap_err("serializing landmarks to a file")
            }
            Actions::Transform {
                features,
                output_dir,
            } => {
                // Retrieve extracted features
                ensure!(features.exists(), "could not find {}", features.display());
                ensure!(features.is_file(), "{} is not a file", features.display());
                let file = std::fs::File::open(features)
                    .into_diagnostic()
                    .wrap_err("opening features file")?;
                let mut features: Features = serde_json::de::from_reader(file)
                    .into_diagnostic()
                    .wrap_err("deserializing features")?;
                // Retrieve reference face
                let Some((name, faces)) = features.first_feature() else {
                    bail!("couldn't find an image with a valid face");
                };
                let Face(_rect, landmarks) = match faces.as_ref() {
                    [] => {
                        warn!("ignoring {name}: no faces found");
                        return Ok(());
                    }
                    [face] => face,
                    _ => {
                        let n = faces.len();
                        warn!("ignoring {name}: found {n} faces, cannot choose which to stabilize");
                        return Ok(());
                    }
                };
                // Copy reference image unchanged
                copy(&name, &features.basedir, &output_dir)?;
                // Transform images
                transform_images(landmarks.clone(), &mut features, &output_dir)
            }
        }
    }
}

#[derive(Debug, Subcommand)]
enum Actions {
    /// Stabilize images
    ///
    /// Uses the first image (alphabetically sorted) as a reference
    Stabilize {
        /// Path to the Shape Predictor model (also called Facial Landmarks Predictor)
        #[arg(env, short, long)]
        shape_predictor: PathBuf,
        /// Path to a directory containing the images you want to extract the features of
        image_dir: PathBuf,
        /// Directory where to place the transformed images
        #[arg(short, long, default_value = "./out")]
        output_dir: PathBuf,
        /// Path to the output file
        #[arg(short, long, default_value = "landmarks.json")]
        output: PathBuf,
        /// Whether to pretty print the extracted text
        #[arg(short, long)]
        pretty: bool,
        /// Path to the CNN face detector model
        #[arg(env, short, long)]
        face_detector: Option<PathBuf>,
        /// Whether to use the CNN based face detector (slower but more accurate)
        #[arg(short = 'c', long)]
        use_cnn_detector: bool,
    },
    /// Extract Features from images to process later
    ExtractFeatures {
        /// Path to the Shape Predictor model (also called Facial Landmarks Predictor)
        #[arg(env, short, long)]
        shape_predictor: PathBuf,
        /// Path to a directory containing the images you want to extract the features of
        image_dir: PathBuf,
        /// Path to the output file
        #[arg(short, long, default_value = "landmarks.json")]
        output: PathBuf,
        /// Whether to pretty print the extracted text
        #[arg(short, long)]
        pretty: bool,
        /// Path to the CNN face detector model
        #[arg(env, short, long)]
        face_detector: Option<PathBuf>,
        /// Whether to use the CNN based face detector (slower but more accurate)
        #[arg(short = 'c', long)]
        use_cnn_detector: bool,
    },
    /// Transform images based on the features extracted previously
    Transform {
        /// Path to the extracted features
        features: PathBuf,
        /// Directory where to place the transformed images
        #[arg(short, long, default_value = "./out")]
        output_dir: PathBuf,
    },
}

fn transform_images(origin: Landmarks, features: &mut Features, output_dir: &Path) -> Result<()> {
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
    let data = features.retrieve_features().into_iter().collect();
    let f = |(name, faces): (Box<str>, Faces)| {
        let img_path = features.feature_path(&name);
        let Face(_rect, landmarks) = match faces.as_ref() {
            [] => {
                warn!("ignoring {name}: no faces found");
                return Ok(());
            }
            [face] => face,
            _ => {
                let n = faces.len();
                warn!("ignoring {name}: found {n} faces, cannot choose which to stabilize");
                return Ok(());
            }
        };
        let img = image::open(&img_path)
            .into_diagnostic()
            .with_context(|| format!("opening image {}", img_path.display()))?
            .into_rgb8();

        let out = out_path(name.as_ref());

        apply_projection(&origin, landmarks, &img, [0, 0, 0].into())
            .save(&out)
            .into_diagnostic()
            .with_context(|| format!("saving image to {}", out.display()))
    };
    maybe_parallel("transforming images", data, ThreadSafe::Yes, |v| Some(f(v)))
}

fn all_file_names_in_dir(path: &Path) -> Result<Vec<Box<str>>> {
    std::fs::read_dir(path)
        .into_diagnostic()
        .wrap_err("trying to open image_dir")?
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
            let fname = dir_ent.file_name();
            let fname = fname.to_string_lossy();
            if !ft.is_file() {
                info!("{fname} is not a file, skipping",);
                return None;
            }
            Some(Ok(fname.into()))
        })
        .collect()
}

fn extract_features(shape_predictor: &Path, basedir: &Path, cnn: bool) -> Result<Features> {
    let file = shape_predictor.display();
    info!("Loading shape predictor from {file}",);
    if !shape_predictor.is_file() {
        bail!("{file} is not a regular file (or doesn't exist).",);
    }
    let predictor = LandmarkPredictor::open(shape_predictor).map_err(|err| miette::miette!(err))?;
    // Get image names
    let imgs = all_file_names_in_dir(basedir)?;
    // Extract features from images
    let features = maybe_parallel::<_, _, Result<_>>("extracting features", imgs, !cnn, |name| {
        Some(extract_feature(basedir, &predictor, cnn)(name))
    })?;
    info!("finished processing");
    Ok(Features {
        basedir: basedir.into(),
        features,
    })
}

fn extract_feature<'a>(
    basedir: &'a Path,
    predictor: &'a LandmarkPredictor,
    cnn: bool,
) -> impl Fn(Box<str>) -> Result<(Box<str>, Faces)> + 'a {
    move |name| {
        let path = basedir.join(name.as_ref());
        let img = image::open(&path)
            .into_diagnostic()
            .with_context(|| format!("failed to open {}", path.display()))?
            .into_rgb8();
        let image = ImageMatrix::from_image(&img);
        let landmarks = if cnn {
            extract_landmarks_cnn(&image, predictor)
        } else {
            extract_landmarks_fast(&image, predictor)
        };
        Ok((name, landmarks))
    }
}

pub type Feature = (Box<str>, Faces);

#[derive(Debug, Serialize, Deserialize)]
pub struct Features {
    /// The directory with all the images
    basedir: Box<Path>,
    /// Mapping from Filename -> Faces
    features: BTreeMap<Box<str>, Faces>,
}

impl Features {
    //pub fn pop_feature(&mut self, key: &str) -> Option<Feature> {
    //    self.features.remove_entry(key)
    //}

    pub fn first_feature(&mut self) -> Option<Feature> {
        self.features.pop_first()
    }

    pub fn retrieve_features(&mut self) -> BTreeMap<Box<str>, Faces> {
        std::mem::take(&mut self.features)
    }

    pub fn feature_path(&self, name: &str) -> PathBuf {
        self.basedir.join(name)
    }
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

fn default_bar_style() -> ProgressStyle {
    ProgressStyle::with_template("[{pos:>4}/{len:4}] {msg} {bar} [{per_sec} {eta_precise}]")
        .expect("valid template")
}

#[cfg(feature = "rayon")]
fn vec_iter<T: Send>(vec: Vec<T>) -> rayon::vec::IntoIter<T> {
    vec.into_par_iter()
}

#[cfg(not(feature = "rayon"))]
fn vec_iter<T: Send>(vec: Vec<T>) -> std::vec::IntoIter<T> {
    vec_iter_seq
}

fn vec_iter_seq<T: Send>(vec: Vec<T>) -> std::vec::IntoIter<T> {
    vec.into_iter()
}

#[cfg(feature = "rayon")]
trait FromMaybeParallel<T: Send>: FromParallelIterator<T> + FromIterator<T> {}
#[cfg(not(feature = "rayon"))]
trait Extendable<T: Send>: FromIterator<T> {}

#[cfg(feature = "rayon")]
impl<T, C> FromMaybeParallel<T> for C
where
    T: Send,
    C: FromParallelIterator<T> + FromIterator<T>,
{
}

#[cfg(not(feature = "rayon"))]
impl<T, C> FromMaybeParallel<T> for C
where
    T: Send,
    C: FromIterator<T>,
{
}

enum ThreadSafe {
    Yes,
    No,
}

impl From<bool> for ThreadSafe {
    fn from(value: bool) -> Self {
        match value {
            true => ThreadSafe::Yes,
            false => ThreadSafe::No,
        }
    }
}

fn maybe_parallel<T, U, C>(
    message: impl Into<Cow<'static, str>>,
    vec: Vec<T>,
    thread_safe: impl Into<ThreadSafe>,
    f: impl Fn(T) -> Option<U> + Send + Sync,
) -> C
where
    T: Send,
    U: Send,
    C: FromMaybeParallel<U>,
{
    match thread_safe.into() {
        ThreadSafe::Yes => vec_iter(vec)
            .progress_with_style(default_bar_style())
            .with_message(message)
            .filter_map(f)
            .collect(),
        ThreadSafe::No => vec_iter_seq(vec)
            .progress_with_style(default_bar_style())
            .filter_map(f)
            .collect(),
    }
}

fn stabilize_image<'a>(
    image_dir: &'a Path,
    output_dir: &'a Path,
    use_cnn: bool,
    predictor: &'a LandmarkPredictor,
    landmarks: &'a Landmarks,
) -> impl Fn(Box<str>) -> Result<Option<(Box<str>, Faces)>> + 'a {
    move |name| {
        let path = image_dir.join(name.as_ref());
        let out = output_dir.join(name.as_ref());
        let img = image::open(&path)
            .into_diagnostic()
            .wrap_err_with(|| format!("failed to open {}", path.display()))?
            .into_rgb8();
        let image = ImageMatrix::from_image(&img);

        let faces = if use_cnn {
            extract_landmarks_cnn(&image, predictor)
        } else {
            extract_landmarks_fast(&image, predictor)
        };
        let Face(_rect, points) = match faces.as_ref() {
            [] => {
                warn!("ignoring {name}: no faces found");
                return Ok(None);
            }
            [face] => face,
            _ => {
                let n = faces.len();
                warn!("ignoring {name}: found {n} faces, cannot choose which to stabilize");
                return Ok(None);
            }
        };
        apply_projection(landmarks, points, &img, [0, 0, 0].into())
            .save(&out)
            .into_diagnostic()
            .wrap_err_with(|| format!("failed to save stabilized image to {}", out.display()))?;
        Ok(Some((name, faces)))
    }
}

fn copy(name: &str, src_dir: &Path, dst_dir: &Path) -> Result<()> {
    let from = src_dir.join(name);
    let to = dst_dir.join(name);
    std::fs::copy(&from, &to)
        .into_diagnostic()
        .wrap_err_with(|| {
            format!(
                "failed to copy file from {} to {}",
                from.display(),
                to.display()
            )
        })
        .map(drop)
}
