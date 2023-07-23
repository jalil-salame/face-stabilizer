use anyhow::anyhow;
use anyhow::bail;
use anyhow::Context;
use clap::Parser;
use dlib_face_recognition::FaceDetector;
use dlib_face_recognition::FaceDetectorTrait;
use dlib_face_recognition::FaceLandmarks;
use dlib_face_recognition::ImageMatrix;
use dlib_face_recognition::LandmarkPredictor;
use dlib_face_recognition::LandmarkPredictorTrait;
use dlib_face_recognition::Rectangle;
use imageproc::geometric_transformations::warp;
use imageproc::geometric_transformations::Interpolation;
use imageproc::geometric_transformations::Projection;
use log::debug;
use log::error;
use log::info;
use log::warn;

#[derive(Debug, Parser)]
struct Opts {
    /// Path to the Shape Predictor model (also called Facial Landmarks Predictor)
    #[arg(env, short, long)]
    shape_predictor: std::path::PathBuf,
    /// Images of faces to stabilize
    #[arg(num_args(1..))]
    images: Vec<std::path::PathBuf>,
    /// Directory to place the stabilized images in
    #[arg(long, short, default_value = "./out")]
    out_dir: std::path::PathBuf,
    /// Reference image (if none is provided the first of the images will be used)
    #[arg(short, long)]
    reference: Option<std::path::PathBuf>,
}

fn main() -> anyhow::Result<()> {
    // Configure using RUST_LOG=* (ie. RUST_LOG=info)
    env_logger::init();

    let mut opts = Opts::parse();
    debug!("Recieved {opts:?}");

    if !opts.out_dir.exists() {
        info!(
            "{dir} does not exist, creating it",
            dir = opts.out_dir.display()
        );
        std::fs::create_dir(&opts.out_dir)
            .with_context(|| format!("Failed to create {}", opts.out_dir.display()))?;
    }

    let reference = opts.reference.unwrap_or_else(|| opts.images.swap_remove(0));

    if !opts.shape_predictor.is_file() {
        bail!(
            "{file} is not a regular file.",
            file = opts.shape_predictor.display()
        );
    }

    let detector = FaceDetector::new();

    info!(
        "Loading shape predictor from {file}",
        file = opts.shape_predictor.display()
    );
    let predictor = LandmarkPredictor::open(opts.shape_predictor).map_err(|err| anyhow!(err))?;

    // Extract landmarks for reference image
    let reference_landmarks =
        extract_landmarks(&img_mat_from_path(&reference)?, &detector, &predictor)
            .with_context(|| format!("failed to get landmarks for reference image"))?;

    let out_path = |path: &std::path::Path| opts.out_dir.join(path.file_name().unwrap());
    let copy_to_out_dir = |path: &std::path::Path| std::fs::copy(path, out_path(path));

    // Save reference to out dir
    copy_to_out_dir(&reference).context("failed to copy file to output directory")?;

    // Get image paths
    let img_paths = {
        let mut imgs = opts.images;
        imgs.sort_unstable();
        imgs
    };

    // Extract landmarks
    for img_path in img_paths {
        match stabilize(&img_path, &reference_landmarks, &detector, &predictor) {
            Err(err) => warn!("{err}"),
            Ok(image) => {
                if let Err(err) = image
                    .save(out_path(&img_path))
                    .context("failed to save image")
                {
                    error!("{err}");
                }
            }
        };
    }

    Ok(())
}

fn stabilize(
    img_path: &std::path::Path,
    reference: &FaceLandmarks,
    detector: &FaceDetector,
    predictor: &LandmarkPredictor,
) -> anyhow::Result<image::RgbImage> {
    let image = image::open(&img_path)?.into_rgb8();
    let img_mat = ImageMatrix::from_image(&image);

    let landmarks = extract_landmarks(&img_mat, detector, predictor)
        .with_context(|| format!("while processing {}", img_path.display()))?;

    let from = points(reference);
    let to = points(&landmarks);

    for (from, to) in choose(from).into_iter().zip(choose(to)) {
        let Some(proj) = Projection::from_control_points(from, to) else { continue; };
        return Ok(warp(
            &image,
            &proj,
            Interpolation::Bicubic,
            image::Rgb([0, 0, 0]),
        ));
    }

    bail!(
        "failed to find a valid projection for {}",
        img_path.display()
    )
}

/// Returns [nose tip, left eye, left eye, right eye, right eye]
fn points(landmarks: &FaceLandmarks) -> [(f32, f32); 5] {
    [
        (landmarks[34 - 1].x() as f32, landmarks[34 - 1].y() as f32),
        (landmarks[37 - 1].x() as f32, landmarks[37 - 1].y() as f32),
        (landmarks[40 - 1].x() as f32, landmarks[40 - 1].y() as f32),
        (landmarks[43 - 1].x() as f32, landmarks[43 - 1].y() as f32),
        (landmarks[46 - 1].x() as f32, landmarks[46 - 1].y() as f32),
    ]
}

fn choose(points: [(f32, f32); 5]) -> [[(f32, f32); 4]; 4] {
    [
        // [points[1], points[2], points[3], points[4]],
        [points[0], points[2], points[3], points[4]],
        [points[0], points[1], points[3], points[4]],
        [points[0], points[1], points[2], points[4]],
        [points[0], points[1], points[2], points[3]],
    ]
}

fn img_mat_from_path(img_path: &std::path::Path) -> image::ImageResult<ImageMatrix> {
    let image = image::open(&img_path)?.into_rgb8();
    Ok(ImageMatrix::from_image(&image))
}

// TODO: Switch to custom result so I can handle the cases with multiple faces
fn extract_landmarks(
    image: &ImageMatrix,
    detector: &FaceDetector,
    predictor: &LandmarkPredictor,
) -> anyhow::Result<FaceLandmarks> {
    let faces = detector.face_locations(&image);
    let faces: &[Rectangle] = &faces;

    let face = match faces {
        [] => {
            bail!("No face found");
        }
        [face] => face,
        _ => {
            bail!("Multiple faces found");
        }
    };

    info!("Found face {face:?}");
    Ok(predictor.face_landmarks(&image, &face))
}
