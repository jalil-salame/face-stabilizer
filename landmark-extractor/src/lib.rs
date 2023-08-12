use dlib_face_recognition::FaceDetectorTrait;
use dlib_face_recognition::FaceLandmarks;
use dlib_face_recognition::ImageMatrix;
use dlib_face_recognition::LandmarkPredictorTrait;
use dlib_face_recognition::Point;
use dlib_face_recognition::Rectangle;

/// Any number of [`Face`]s
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Faces(Box<[Face]>);

impl std::ops::Deref for Faces {
    type Target = [Face];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl From<Faces> for Box<[Face]> {
    fn from(value: Faces) -> Self {
        value.0
    }
}

/// The bounding box ([`Rect`]) and [`Landmarks`] of a face
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Face(pub Rect, pub Landmarks);

impl From<Face> for (Rectangle, Landmarks) {
    fn from(value: Face) -> Self {
        (value.0.into(), value.1)
    }
}

/// A bounding box
///
/// Derives serde traits, unlike the [`dlib_face_recognition::Rectangle`]
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Rect {
    pub left: i64,
    pub top: i64,
    pub right: i64,
    pub bottom: i64,
}

impl From<Rect> for Rectangle {
    fn from(value: Rect) -> Self {
        let Rect {
            left,
            top,
            right,
            bottom,
        } = value;

        Self {
            left,
            top,
            right,
            bottom,
        }
    }
}

impl From<Rectangle> for Rect {
    fn from(value: Rectangle) -> Self {
        let Rectangle {
            left,
            top,
            right,
            bottom,
        } = value;

        Self {
            left,
            top,
            right,
            bottom,
        }
    }
}

/// Facial Landmarks
///
/// Derives more traits unlike [`dlib_face_recognition::FaceLandmarks`]
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Landmarks(Box<[(i64, i64)]>);

impl std::ops::Deref for Landmarks {
    type Target = [(i64, i64)];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl From<Landmarks> for Box<[Point]> {
    fn from(value: Landmarks) -> Self {
        value
            .0
            .into_vec()
            .into_iter()
            .map(|(x, y)| Point::new(x, y))
            .collect()
    }
}

impl From<FaceLandmarks> for Landmarks {
    fn from(value: FaceLandmarks) -> Self {
        Self(value.to_vec().into_iter().map(|p| (p.x(), p.y())).collect())
    }
}

/// Find all faces in this image and identify the landmarks in it
pub fn extract_landmarks(
    image: &ImageMatrix,
    detector: &impl FaceDetectorTrait,
    predictor: &impl LandmarkPredictorTrait,
) -> Faces {
    let landmarks = detector
        .face_locations(image)
        .iter()
        .cloned()
        .map(|face| Face(face.into(), predictor.face_landmarks(image, &face).into()))
        .collect();

    Faces(landmarks)
}

/// Helper function to load an [`ImageMatrix`] from a path
#[cfg(feature = "image")]
pub fn img_mat_from_path(img_path: &std::path::Path) -> image::ImageResult<ImageMatrix> {
    let image = image::open(&img_path)?.into_rgb8();
    Ok(ImageMatrix::from_image(&image))
}
