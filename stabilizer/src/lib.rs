use glam::Vec2;
use imageproc::geometric_transformations::Projection;

/// Calculates the "center of mass" of a set of points
///
/// Returns [`None`] if empty
pub fn centroid(points: &[Vec2]) -> Option<Vec2> {
    if points.is_empty() {
        return None;
    }
    // Mean
    Some(points.iter().sum::<Vec2>() / points.len() as f32)
}

/// "Center" all of the points and return the centroid
///
/// Returns [`None`] if empty
pub fn center(points: &mut [Vec2]) -> Option<Vec2> {
    let c = centroid(points)?;
    for point in points {
        *point -= c;
    }
    Some(c)
}

/// Compute the what the "scaling factor" would be if on average all points where 1 unit away from
/// the origin.
///
/// Expects the points to be "centered" (the centroid should be at the origin), you can ensure this
/// by calling [`center`] first
///
/// Returns [`None`] if empty
pub fn scaling_factor(points: &[Vec2]) -> Option<f32> {
    if points.is_empty() {
        return None;
    }
    // √((||p₁||² + ||p₂||² + ...) / points.len())
    Some((points.iter().map(|p| p.length_squared()).sum::<f32>() / points.len() as f32).sqrt())
}

/// Scale the points down so that on average all points are 1 unit away from the origin.
///
/// Expects the points to be "centered" (the centroid should be at the origin), you can ensure this
/// by calling [`center`] first
///
/// Returns [`None`] if empty
pub fn scale(points: &mut [Vec2]) -> Option<f32> {
    let s = scaling_factor(points)?;
    for point in points {
        *point /= s;
    }
    Some(s)
}

/// Calculate the angle of rotation that reduces the error between [`reference`] and [`points`]. The
/// resulting angle will be between [-π/2, π/2] radians.
///
/// Expects both the reference and the points to be centered and scaled, use [`center`] and
/// [`scale`] to achieve this.
///
/// Returns [`None`] if empty
pub fn rotation(reference: &[Vec2], points: &[Vec2]) -> Option<f32> {
    if points.is_empty() || (points.len() != reference.len()) {
        return None;
    }
    let top: f32 = points
        .iter()
        .zip(reference)
        .map(|(p, r)| p.x * r.y - p.y * r.x)
        .sum();
    let bot: f32 = points
        .iter()
        .zip(reference)
        .map(|(p, r)| p.x * r.x + p.y * r.y)
        .sum();
    Some((top / bot).atan())
}

/// Calculate the [`Projection`] that better approximates the shapes of the points.
/// See the [wikipedia](https://en.wikipedia.org/wiki/Procrustes_analysis) page
///
/// [`target`] is what you want the end result to be [`points`] are the points that will be
/// transformed.
///
/// Returns [`None`] if empty
pub fn procrustes_superimposition(target: &mut [Vec2], points: &mut [Vec2]) -> Option<Projection> {
    // Calculate translation vector
    let tt = center(target)?;
    let pt = center(points)?;
    // let t = tt - pt;
    // Calculate the scale
    let ts = scale(target)?;
    let ps = scale(points)?;
    // let s = ts / ps;
    // Calculate rotation
    let theta = rotation(target, points)?;
    // Create Projection
    Some(
        Projection::translate(-pt.x, -pt.y)
            .and_then(Projection::scale(1.0 / ps, 1.0 / ps))
            .and_then(Projection::rotate(theta))
            .and_then(Projection::scale(ts, ts))
            .and_then(Projection::translate(tt.x, tt.y)),
    )
}
