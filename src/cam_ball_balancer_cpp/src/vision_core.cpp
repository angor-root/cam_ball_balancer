#include "cam_ball_balancer_cpp/vision_core.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

namespace cbb {

// ================================================================ Kalman
Kalman2D::Kalman2D(KalmanConfig cfg) : cfg_(cfg), x_(0, 0, 0, 0), P_(cv::Matx44d::eye()) {}

void Kalman2D::reset(double x, double y) {
  x_ = {x, y, 0, 0};
  P_ = cv::Matx44d::diag({cfg_.initial_pos_var, cfg_.initial_pos_var,
                          cfg_.initial_vel_var, cfg_.initial_vel_var});
  initialized = true;
}

void Kalman2D::predict(double dt) {
  if (!initialized) return;
  cv::Matx44d F = cv::Matx44d::eye();
  F(0, 2) = dt;
  F(1, 3) = dt;
  const double d = std::max(dt, 1e-6);
  const double qp = cfg_.process_noise_pos * d, qv = cfg_.process_noise_vel * d;
  x_ = F * x_;
  P_ = F * P_ * F.t() + cv::Matx44d::diag({qp, qp, qv, qv});
}

void Kalman2D::update(double zx, double zy) {
  if (!initialized) {
    reset(zx, zy);
    return;
  }
  // H = [I2 0]: S = P[0:2,0:2] + R, K = P[:,0:2] S^-1 (sin matrices H explicitas).
  const double r = cfg_.measurement_noise;
  cv::Matx22d S(P_(0, 0) + r, P_(0, 1), P_(1, 0), P_(1, 1) + r);
  cv::Matx22d Si = S.inv();
  cv::Matx<double, 4, 2> PHt;
  for (int i = 0; i < 4; ++i) { PHt(i, 0) = P_(i, 0); PHt(i, 1) = P_(i, 1); }
  cv::Matx<double, 4, 2> K = PHt * Si;
  cv::Vec2d y(zx - x_(0), zy - x_(1));
  cv::Vec4d dx = K * y;
  x_ += dx;
  cv::Matx<double, 2, 4> HP;
  for (int j = 0; j < 4; ++j) { HP(0, j) = P_(0, j); HP(1, j) = P_(1, j); }
  P_ = P_ - K * HP;
}

// ================================================================ Pelota
BallDetector::BallDetector(BallDetectorConfig c) : cfg(c) {
  kernel_ = cv::Mat::ones(cfg.morph_kernel, cfg.morph_kernel, CV_8U);
}

std::optional<Detection> BallDetector::detect(const cv::Mat& frame) {
  const int f = std::max(cfg.downscale, 1);
  const cv::Mat* src = &frame;
  if (f > 1) {
    cv::resize(frame, small_, cv::Size(), 1.0 / f, 1.0 / f, cv::INTER_AREA);
    src = &small_;
  }
  // kernel <= 1 desactiva el filtro (cada uno es tuneable por costo en la Pi).
  const cv::Mat* cur = src;
  if (cfg.median_kernel > 1) { cv::medianBlur(*cur, denoised_, cfg.median_kernel); cur = &denoised_; }
  if (cfg.blur_kernel > 1) {
    cv::GaussianBlur(*cur, blurred_, {cfg.blur_kernel, cfg.blur_kernel}, 0);
    cur = &blurred_;
  }
  cv::cvtColor(*cur, hsv_, cv::COLOR_BGR2HSV);
  cv::inRange(hsv_, cfg.hsv_lower, cfg.hsv_upper, mask_);
  cv::morphologyEx(mask_, mask_, cv::MORPH_OPEN, kernel_);
  cv::morphologyEx(mask_, mask_, cv::MORPH_CLOSE, kernel_);

  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(mask_, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
  std::optional<Detection> best;
  double best_area = 0;
  for (const auto& c : contours) {
    const double area = cv::contourArea(c);
    if (area <= 0) continue;
    cv::Point2f center;
    float radius;
    cv::minEnclosingCircle(c, center, radius);
    if (radius * f < cfg.min_radius_px || radius * f > cfg.max_radius_px) continue;
    const double circ = area / (CV_PI * radius * radius);
    if (circ < cfg.min_circularity) continue;
    if (area > best_area) {
      best_area = area;
      // +0.5*(f-1): el pixel (i) de la imagen reducida cubre [f*i, f*i+f-1].
      best = Detection{center.x * f + 0.5 * (f - 1), center.y * f + 0.5 * (f - 1), radius * f};
    }
  }
  return best;
}

const char* to_string(TrackState s) {
  switch (s) {
    case TrackState::Searching: return "searching";
    case TrackState::Tracking: return "tracking";
    case TrackState::Coasting: return "coasting";
    case TrackState::Lost: return "lost";
    case TrackState::NoPose: return "no_pose";
  }
  return "?";
}

// ================================================================ Camara
Intrinsics Intrinsics::load(const std::string& path) {
  cv::FileStorage fs(path, cv::FileStorage::READ);
  if (!fs.isOpened()) throw std::runtime_error("No se pudo abrir " + path);
  cv::Mat K, d;
  fs["camera_matrix"] >> K;
  fs["dist_coeffs"] >> d;
  if (K.empty()) throw std::runtime_error("camera_matrix vacia en " + path);
  Intrinsics in;
  in.K = cv::Matx33d(K);
  in.dist = d.reshape(1, 1).clone();
  return in;
}

Intrinsics Intrinsics::guess(int w, int h) {
  Intrinsics in;
  in.K = cv::Matx33d(w, 0, w / 2.0, 0, w, h / 2.0, 0, 0, 1);
  in.dist = cv::Mat::zeros(1, 5, CV_64F);
  return in;
}

// ================================================================ Pose
PoseEstimator::PoseEstimator(PoseConfig c, Intrinsics intr) : cfg(c), intr_(std::move(intr)) {
  dict_ = cv::aruco::getPredefinedDictionary(cfg.dictionary);
  params_ = cv::aruco::DetectorParameters::create();
  params_->adaptiveThreshWinSizeMin = cfg.thresh_win_min;
  params_->adaptiveThreshWinSizeMax = cfg.thresh_win_max;
  params_->adaptiveThreshWinSizeStep = cfg.thresh_win_step;
  params_->minMarkerPerimeterRate = cfg.min_marker_perimeter_rate;
}

void tilt_from_rvec(const cv::Vec3d& rvec, double& theta_x, double& theta_y) {
  cv::Matx33d R;
  cv::Rodrigues(rvec, R);
  // Normal del plato (su +Z, hacia la camara) en el frame optico. Plato
  // plano => n = (0, 0, -1). Se expresa en el frame "plano de referencia"
  // F = Rx(pi) * optico (x igual, y y z invertidos): n_F = (n0, -n1, -n2),
  // y ahi theta_x = atan2(-n_F1, n_F2), theta_y = atan2(n_F0, n_F2):
  // rotacion de mano derecha del plato sobre su propio X / Y.
  const cv::Vec3d n(R(0, 2), R(1, 2), R(2, 2));
  theta_x = std::atan2(n[1], -n[2]);
  theta_y = std::atan2(n[0], -n[2]);
}

PlatePose PoseEstimator::estimate(const cv::Mat& gray, const cv::Rect& search) {
  PlatePose out;
  corners_.clear();
  ids_.clear();
  const cv::Rect area = search.area() > 0 ? search : cv::Rect(0, 0, gray.cols, gray.rows);
  const int f = std::max(cfg.aruco_downscale, 1);
  if (f > 1) {
    cv::resize(gray(area), small_, cv::Size(), 1.0 / f, 1.0 / f, cv::INTER_AREA);
    cv::aruco::detectMarkers(small_, dict_, corners_, ids_, params_, rejected_);
    if (!ids_.empty()) {
      // centro del pixel reducido i = f*i + (f-1)/2 en coordenadas continuas: x = (x_s + 0.5)*f - 0.5
      std::vector<cv::Point2f> pts;
      for (const auto& mk : corners_)
        for (const auto& p : mk) pts.emplace_back((p.x + 0.5f) * f - 0.5f, (p.y + 0.5f) * f - 0.5f);
      cv::cornerSubPix(gray(area), pts, cv::Size(f + 1, f + 1), cv::Size(-1, -1),
                       cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 10, 0.05));
      size_t i = 0;
      for (auto& mk : corners_)
        for (auto& p : mk) p = pts[i++];
    }
  } else {
    cv::aruco::detectMarkers(gray(area), dict_, corners_, ids_, params_, rejected_);
  }
  auto fail = [&]() {
    if (++frames_without_pose_ > cfg.max_frames_without_pose) prev_rvec_.reset();
    out.corner_px.clear();
    return out;
  };
  if (ids_.empty()) return fail();
  if (area.x || area.y)
    for (auto& mk : corners_)
      for (auto& p : mk) { p.x += area.x; p.y += area.y; }

  const double h = cfg.marker_length_m / 2.0;
  std::vector<cv::Point3d> obj;
  std::vector<cv::Point2d> img;
  for (size_t k = 0; k < ids_.size(); ++k) {
    auto it = std::find(cfg.corner_ids.begin(), cfg.corner_ids.end(), ids_[k]);
    if (it == cfg.corner_ids.end()) continue;
    const cv::Point2d c = cfg.corner_pos_m[it - cfg.corner_ids.begin()];
    // Esquinas TL, TR, BR, BL del propio tag tal como las devuelve ArUco
    // (TL = arriba-izquierda en la imagen). Frame del plato REP-103:
    // X derecha, Y "arriba" en la imagen, Z saliendo del plato hacia la
    // camara -> "arriba" = +y.
    obj.push_back({c.x - h, c.y + h, 0});
    obj.push_back({c.x + h, c.y + h, 0});
    obj.push_back({c.x + h, c.y - h, 0});
    obj.push_back({c.x - h, c.y - h, 0});
    cv::Point2f center(0, 0);
    for (const auto& p : corners_[k]) { img.emplace_back(p); center += p * 0.25f; }
    out.corner_px.push_back(center);
  }
  if (obj.size() < 8) return fail();  // < 2 tags

  std::vector<cv::Mat> rvecs, tvecs;
  cv::Mat errors;
  const int n = cv::solvePnPGeneric(obj, img, intr_.K, intr_.dist, rvecs, tvecs, false,
                                    cv::SOLVEPNP_IPPE, cv::noArray(), cv::noArray(), errors);
  if (n == 0) return fail();
  int best = 0;
  for (int i = 1; i < n; ++i)
    if (errors.at<double>(i) < errors.at<double>(best)) best = i;
  int pick = best;
  if (n == 2) {
    const int other = 1 - best;
    out.ambiguity = errors.at<double>(other) / std::max(errors.at<double>(best), 1e-9);
    if (out.ambiguity < cfg.ambiguity_ratio && prev_rvec_) {
      // Empate: la solucion real es la continua con el cuadro anterior.
      double tx0, ty0, tx1, ty1, txp, typ;
      tilt_from_rvec(cv::Vec3d(rvecs[best]), tx0, ty0);
      tilt_from_rvec(cv::Vec3d(rvecs[other]), tx1, ty1);
      tilt_from_rvec(*prev_rvec_, txp, typ);
      if (std::hypot(tx1 - txp, ty1 - typ) < std::hypot(tx0 - txp, ty0 - typ)) {
        pick = other;
        out.disambiguated = true;
      }
    }
  }
  cv::Mat rv = rvecs[pick].clone(), tv = tvecs[pick].clone();
  // Salvaguarda: IPPE de OpenCV degenera con el plato EXACTAMENTE de frente
  // (devuelve basura con ~180 px de error; con 0.001 rad ya funciona). Si la
  // solucion es absurda (error enorme o plato mirando hacia atras, imposible
  // porque los tags no se ven por detras) se resuelve con SQPnP (global).
  auto facing_camera = [](const cv::Mat& r, const cv::Mat& t) {
    cv::Matx33d R;
    cv::Rodrigues(r, R);
    const cv::Vec3d n(R(0, 2), R(1, 2), R(2, 2));
    return n.dot(cv::Vec3d(t)) < 0;  // +Z del plato apunta hacia la camara
  };
  if (errors.at<double>(pick) > cfg.max_reproj_err_px || !facing_camera(rv, tv)) {
    if (!cv::solvePnP(obj, img, intr_.K, intr_.dist, rv, tv, false, cv::SOLVEPNP_SQPNP) ||
        !facing_camera(rv, tv))
      return fail();
  }
  // Refinamiento Levenberg-Marquardt sobre todos los puntos (menos ruido
  // entre cuadros; medido offline: 0.58 -> 0.24 grados tipicos).
  cv::solvePnPRefineLM(obj, img, intr_.K, intr_.dist, rv, tv);
  out.rvec = cv::Vec3d(rv);
  out.tvec = cv::Vec3d(tv);
  {
    std::vector<cv::Point2d> proj;
    cv::projectPoints(obj, rv, tv, intr_.K, intr_.dist, proj);
    double se = 0;
    for (size_t i = 0; i < proj.size(); ++i) {
      const cv::Point2d d = proj[i] - img[i];
      se += d.dot(d);
    }
    out.reproj_err_px = std::sqrt(se / proj.size());
  }
  tilt_from_rvec(out.rvec, out.theta_x, out.theta_y);
  out.valid = true;
  prev_rvec_ = out.rvec;
  frames_without_pose_ = 0;
  return out;
}

// ================================================================ Rayo-plano
PlateRayMapper::PlateRayMapper(Intrinsics intr, double r) : intr_(std::move(intr)), r_(r) {}

void PlateRayMapper::set_pose(const cv::Vec3d& rvec, const cv::Vec3d& tvec) {
  cv::Rodrigues(rvec, R_);
  t_ = tvec;
  ready_ = true;
}

bool PlateRayMapper::to_plane(double u, double v, double& x, double& y) const {
  if (!ready_) return false;
  std::vector<cv::Point2d> src{{u, v}}, dst;
  cv::undistortPoints(src, dst, intr_.K, intr_.dist);
  const cv::Vec3d d(dst[0].x, dst[0].y, 1.0);
  const cv::Vec3d n(R_(0, 2), R_(1, 2), R_(2, 2));
  const cv::Vec3d p0 = t_ + R_ * cv::Vec3d(0, 0, r_);  // centro de la pelota: +r sobre el plato
  const double denom = n.dot(d);
  if (std::abs(denom) < 1e-9) return false;
  const double s = n.dot(p0) / denom;
  const cv::Vec3d p = R_.t() * (s * d - t_);
  x = p[0];
  y = p[1];
  return true;
}

// ================================================================ Tracker
BallTracker::BallTracker(BallDetectorConfig det, KalmanConfig kf, int max_misses,
                         double gms, double gmj)
    : detector(det), kf_(kf), max_misses_(max_misses), gate_max_speed_(gms), gate_min_jump_(gmj) {}

BallMeasurement BallTracker::update(const cv::Mat& frame, double t, const cv::Rect& roi,
                                    const PlateRayMapper& mapper) {
  const double dt = last_t_ ? std::max(t - *last_t_, 0.0) : 0.0;
  last_t_ = t;
  kf_.predict(dt);

  const cv::Rect area = roi.area() > 0 ? roi : cv::Rect(0, 0, frame.cols, frame.rows);
  auto det = detector.detect(frame(area));
  bool accepted = false;
  if (det) {
    det->u += area.x;
    det->v += area.y;
    double x, y;
    if (mapper.to_plane(det->u, det->v, x, y)) {
      if (!kf_.initialized) {
        kf_.update(x, y);
        accepted = true;
      } else {
        const double jump = std::hypot(x - kf_.px(), y - kf_.py());
        if (jump <= gate_min_jump_ + gate_max_speed_ * dt) {
          kf_.update(x, y);
          accepted = true;
        }
      }
    }
  }
  misses_ = accepted ? 0 : misses_ + 1;

  if (!kf_.initialized) {
    state = accepted ? TrackState::Tracking : TrackState::Searching;
  } else if (accepted) {
    state = TrackState::Tracking;
  } else if (misses_ > max_misses_) {
    state = TrackState::Lost;
    kf_.initialized = false;
  } else {
    state = TrackState::Coasting;
  }

  BallMeasurement m;
  m.t = t;
  m.state = state;
  if (!kf_.initialized) return m;
  m.x = kf_.px();
  m.y = kf_.py();
  m.valid = state != TrackState::Lost;
  if (accepted) {
    m.pixel = cv::Point2d(det->u, det->v);
    m.radius_px = det->r;
  }
  return m;
}

// ================================================================ Tasa
void RateStats::tick(double t) {
  std::lock_guard<std::mutex> lk(mtx_);
  if (last_) {
    periods_.push_back(t - *last_);
    if (periods_.size() > window_) periods_.pop_front();
  }
  last_ = t;
}

RateStats::Summary RateStats::summary() const {
  std::lock_guard<std::mutex> lk(mtx_);
  Summary s;
  if (periods_.empty()) return s;
  double sum = 0, mx = 0;
  for (double p : periods_) { sum += p; mx = std::max(mx, p); }
  const double mean = sum / periods_.size();
  double var = 0;
  for (double p : periods_) var += (p - mean) * (p - mean);
  s.hz = mean > 0 ? 1.0 / mean : 0;
  s.period_ms = mean * 1e3;
  s.jitter_ms = std::sqrt(var / periods_.size()) * 1e3;
  s.period_max_ms = mx * 1e3;
  return s;
}

void RateStats::reset() {
  std::lock_guard<std::mutex> lk(mtx_);
  periods_.clear();
  last_.reset();
}

// ================================================================ Pipeline
VisionPipeline::VisionPipeline(PipelineConfig c, Intrinsics intr)
    : cfg(c), intr_(intr), pose_est_(c.pose, intr), mapper_(intr, c.ball_radius_m),
      tracker_(c.detector, c.kalman, c.max_consecutive_misses, c.gate_max_speed, c.gate_min_jump) {}

void VisionPipeline::update_roi(const PlatePose& p, cv::Size size) {
  if (!cfg.use_roi) { roi_ = {}; return; }
  const double h = cfg.pose.marker_length_m / 2.0;
  std::vector<cv::Point3d> pts;
  for (const auto& c : cfg.pose.corner_pos_m)
    for (int sx : {-1, 1})
      for (int sy : {-1, 1}) pts.push_back({c.x + sx * h, c.y + sy * h, 0});
  std::vector<cv::Point2d> img;
  cv::projectPoints(pts, p.rvec, p.tvec, intr_.K, intr_.dist, img);
  std::vector<cv::Point2f> imgf(img.begin(), img.end());  // boundingRect no acepta double
  const cv::Rect bb = cv::boundingRect(imgf);
  const int m = cfg.roi_margin_px;
  cv::Rect r(cv::Point(bb.x - m, bb.y - m), cv::Point(bb.x + bb.width + m, bb.y + bb.height + m));
  roi_ = r & cv::Rect(0, 0, size.width, size.height);
}

VisionResult VisionPipeline::process(const cv::Mat& bgr, double t) {
  VisionResult res;
  res.t = t;

  const bool run_aruco = !last_pose_ || frame_idx_ % std::max(cfg.aruco_every_n, 1) == 0;
  if (run_aruco) {
    const double t0 = now_s();
    cv::cvtColor(bgr, gray_, cv::COLOR_BGR2GRAY);
    PlatePose p;
    if (cfg.use_roi && roi_.area() > 0) {
      const int m = cfg.aruco_roi_margin_px;
      const cv::Rect search = cv::Rect(roi_.x - m, roi_.y - m, roi_.width + 2 * m, roi_.height + 2 * m) &
                              cv::Rect(0, 0, bgr.cols, bgr.rows);
      p = pose_est_.estimate(gray_, search);
    }
    if (!p.valid) p = pose_est_.estimate(gray_);
    if (p.valid) {
      last_pose_ = p;
      pose_age_ = 0;
      res.pose_fresh = true;
      mapper_.set_pose(p.rvec, p.tvec);
      update_roi(p, bgr.size());
    }
    res.aruco_ms = (now_s() - t0) * 1e3;
  }
  if (!res.pose_fresh) ++pose_age_;
  if (last_pose_ && pose_age_ <= cfg.pose_hold_max_frames) res.pose = *last_pose_;

  const double t1 = now_s();
  if (mapper_.ready() && res.pose.valid) {
    res.ball = tracker_.update(bgr, t, roi_, mapper_);
  } else {
    res.ball.t = t;
    res.ball.state = TrackState::NoPose;
  }
  res.ball_ms = (now_s() - t1) * 1e3;
  res.roi = roi_;
  ++frame_idx_;
  return res;
}

cv::Mat draw_debug(const cv::Mat& bgr, const VisionResult& r, double hz) {
  cv::Mat out = bgr.clone();
  if (r.roi.area() > 0) cv::rectangle(out, r.roi, {255, 200, 0}, 1);
  for (const auto& c : r.pose.corner_px) cv::circle(out, c, 4, {255, 0, 255}, -1);
  if (r.ball.pixel) cv::circle(out, *r.ball.pixel, int(std::max(r.ball.radius_px, 5.0)), {0, 255, 0}, 2);
  char buf[128];
  std::snprintf(buf, sizeof buf, "tilt x=%+.2f y=%+.2f deg%s", r.pose.theta_x * 180 / CV_PI,
                r.pose.theta_y * 180 / CV_PI, r.pose.valid ? "" : " (INVALID)");
  cv::putText(out, buf, {8, 20}, cv::FONT_HERSHEY_SIMPLEX, 0.5, {255, 255, 255}, 1);
  if (r.ball.valid)
    std::snprintf(buf, sizeof buf, "ball %s: x=%+.1f y=%+.1f cm", to_string(r.ball.state),
                  r.ball.x * 100, r.ball.y * 100);
  else
    std::snprintf(buf, sizeof buf, "ball %s", to_string(r.ball.state));
  cv::putText(out, buf, {8, 40}, cv::FONT_HERSHEY_SIMPLEX, 0.5, {255, 255, 255}, 1);
  std::snprintf(buf, sizeof buf, "%.1f Hz", hz);
  cv::putText(out, buf, {8, 60}, cv::FONT_HERSHEY_SIMPLEX, 0.5, {255, 255, 255}, 1);
  return out;
}

}  // namespace cbb
