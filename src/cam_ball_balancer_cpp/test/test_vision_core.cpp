// Mismos casos que test/test_vision_pipeline.py, sobre el port C++.
#include <gtest/gtest.h>

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

#include "cam_ball_balancer_cpp/vision_core.hpp"

using namespace cbb;

namespace {
constexpr double kBallR = 0.02;

Intrinsics intr() {
  Intrinsics in;
  in.K = cv::Matx33d(700, 0, 320, 0, 700, 240, 0, 0, 1);
  in.dist = cv::Mat::zeros(1, 5, CV_64F);
  return in;
}

// Render con proyeccion pinhole real (equivalente a
// synthetic.render_plate_with_markers_perspective) + pelota naranja.
cv::Mat render(const PoseConfig& pc, const Intrinsics& in, cv::Vec3d rvec, cv::Vec3d tvec,
               std::optional<cv::Point2d> ball) {
  cv::Mat frame(480, 640, CV_8UC3, cv::Scalar(60, 100, 60));
  auto dict = cv::aruco::getPredefinedDictionary(pc.dictionary);
  const double h = pc.marker_length_m / 2;
  const int bs = 200;
  for (int i = 0; i < 4; ++i) {
    const auto c = pc.corner_pos_m[i];
    std::vector<cv::Point3d> obj{{c.x - h, c.y + h, 0}, {c.x + h, c.y + h, 0},
                                 {c.x + h, c.y - h, 0}, {c.x - h, c.y - h, 0}};
    std::vector<cv::Point2d> img;
    cv::projectPoints(obj, rvec, tvec, in.K, in.dist, img);
    cv::Mat marker, marker_bgr;
    cv::aruco::drawMarker(dict, pc.corner_ids[i], bs, marker);
    cv::cvtColor(marker, marker_bgr, cv::COLOR_GRAY2BGR);
    std::vector<cv::Point2f> src{{0, 0}, {bs - 1.f, 0}, {bs - 1.f, bs - 1.f}, {0, bs - 1.f}};
    std::vector<cv::Point2f> dst(img.begin(), img.end());
    cv::Mat H = cv::getPerspectiveTransform(src, dst), warped;
    cv::warpPerspective(marker_bgr, warped, H, frame.size(), cv::INTER_LINEAR,
                        cv::BORDER_CONSTANT, cv::Scalar(60, 100, 60));
    cv::Mat mask = cv::Mat::zeros(frame.size(), CV_8U);
    std::vector<cv::Point> poly(dst.begin(), dst.end());
    cv::fillConvexPoly(mask, poly, 255);
    warped.copyTo(frame, mask);
  }
  if (ball) {
    std::vector<cv::Point3d> c{{ball->x, ball->y, kBallR}};
    std::vector<cv::Point2d> px;
    cv::projectPoints(c, rvec, tvec, in.K, in.dist, px);
    const int r = int(std::round(700 * kBallR / tvec[2]));
    cv::circle(frame, cv::Point(int(std::round(px[0].x)), int(std::round(px[0].y))), r,
               cv::Scalar(0, 140, 255), -1);
  }
  return frame;
}
// rvec (plato -> optico) de un plato con Z hacia la camara, inclinado
// ax sobre su X y luego ay sobre su Y: R = Rx(pi) * Rx(ax) * Ry(ay).
cv::Vec3d plate_rvec(double ax, double ay) {
  cv::Matx33d Rf, Rx, Ry;
  cv::Rodrigues(cv::Vec3d(CV_PI, 0, 0), Rf);
  cv::Rodrigues(cv::Vec3d(ax, 0, 0), Rx);
  cv::Rodrigues(cv::Vec3d(0, ay, 0), Ry);
  cv::Vec3d r;
  cv::Rodrigues(Rf * Rx * Ry, r);
  return r;
}
}  // namespace

TEST(Tilt, AxisConvention) {
  double tx, ty;
  tilt_from_rvec(plate_rvec(0, 0), tx, ty);
  EXPECT_NEAR(tx, 0.0, 1e-6);
  EXPECT_NEAR(ty, 0.0, 1e-6);
  tilt_from_rvec(plate_rvec(0.2, 0), tx, ty);
  EXPECT_NEAR(tx, 0.2, 1e-6);
  EXPECT_NEAR(ty, 0.0, 1e-6);
  tilt_from_rvec(plate_rvec(0, 0.15), tx, ty);
  EXPECT_NEAR(tx, 0.0, 1e-6);
  EXPECT_NEAR(ty, 0.15, 1e-6);
}

TEST(Frame, PlateZPointsToCamera) {
  cv::Matx33d R;
  cv::Rodrigues(plate_rvec(0.1, -0.05), R);
  EXPECT_LT(R(2, 2), -0.9);  // +Z del plato apunta hacia la camara (-Z optico)
}

TEST(RayMapper, InvertsProjectionOnTiltedPlate) {
  const auto in = intr();
  const cv::Vec3d rvec = plate_rvec(0.15, -0.1), tvec(0.01, -0.02, 0.45);
  PlateRayMapper m(in, kBallR);
  m.set_pose(rvec, tvec);
  for (auto p : {cv::Point2d(0, 0), cv::Point2d(0.06, -0.04), cv::Point2d(-0.07, 0.05)}) {
    std::vector<cv::Point3d> c{{p.x, p.y, kBallR}};
    std::vector<cv::Point2d> px;
    cv::projectPoints(c, rvec, tvec, in.K, in.dist, px);
    double x, y;
    ASSERT_TRUE(m.to_plane(px[0].x, px[0].y, x, y));
    EXPECT_NEAR(x, p.x, 1e-6);
    EXPECT_NEAR(y, p.y, 1e-6);
  }
}

TEST(Pipeline, BallInMetersOnTiltedPlate) {
  PipelineConfig pc;
  pc.ball_radius_m = kBallR;
  const cv::Vec3d rvec = plate_rvec(0.12, -0.08), tvec(0, 0, 0.5);
  VisionPipeline pipe(pc, intr());
  const cv::Point2d truth(0.04, -0.03);
  auto r = pipe.process(render(pc.pose, intr(), rvec, tvec, truth), 0.0);
  ASSERT_TRUE(r.pose.valid);
  EXPECT_TRUE(r.pose_fresh);
  EXPECT_NEAR(r.pose.theta_x, 0.12, 0.05);
  EXPECT_NEAR(r.pose.theta_y, -0.08, 0.05);
  ASSERT_TRUE(r.ball.valid);
  EXPECT_NEAR(r.ball.x, truth.x, 0.006);
  EXPECT_NEAR(r.ball.y, truth.y, 0.006);
  EXPECT_GT(r.roi.area(), 0);
}

TEST(Pipeline, NoPoseMeansNoBall) {
  VisionPipeline pipe(PipelineConfig{}, intr());
  cv::Mat blank(480, 640, CV_8UC3, cv::Scalar(60, 100, 60));
  cv::circle(blank, {320, 240}, 14, cv::Scalar(0, 140, 255), -1);
  auto r = pipe.process(blank, 0.0);
  EXPECT_FALSE(r.pose.valid);
  EXPECT_FALSE(r.ball.valid);
  EXPECT_EQ(r.ball.state, TrackState::NoPose);
}

TEST(Pipeline, ArucoEveryNHoldsPose) {
  PipelineConfig pc;
  pc.aruco_every_n = 3;
  VisionPipeline pipe(pc, intr());
  auto frame = render(pc.pose, intr(), plate_rvec(0.1, 0), {0, 0, 0.5}, cv::Point2d(0, 0));
  std::vector<bool> fresh;
  for (int i = 0; i < 6; ++i) fresh.push_back(pipe.process(frame, i / 30.0).pose_fresh);
  EXPECT_EQ(fresh, (std::vector<bool>{true, false, false, true, false, false}));
}

TEST(Tracker, RejectsOutlierJump) {
  PipelineConfig pc;
  VisionPipeline pipe(pc, intr());
  const cv::Vec3d rvec = plate_rvec(0, 0), tvec(0, 0, 0.5);
  for (int i = 0; i < 5; ++i)
    pipe.process(render(pc.pose, intr(), rvec, tvec, cv::Point2d(0.0, 0.0)), i / 30.0);
  // Salto de 10 cm en 1/30 s (3 m/s) > gate (2 m/s*dt + 2 cm = 8.7 cm): se rechaza.
  auto r = pipe.process(render(pc.pose, intr(), rvec, tvec, cv::Point2d(0.10, 0.0)), 5 / 30.0);
  EXPECT_EQ(r.ball.state, TrackState::Coasting);
  EXPECT_NEAR(r.ball.x, 0.0, 0.01);
}

TEST(Pose, ExactlyFrontoParallelIsSane) {
  // Regresion: IPPE degenera en este caso exacto (ver vision_core.cpp).
  PoseConfig pc;
  PoseEstimator est(pc, intr());
  cv::Mat gray;
  cv::cvtColor(render(pc, intr(), plate_rvec(0, 0), {0, 0, 0.5}, std::nullopt), gray, cv::COLOR_BGR2GRAY);
  auto p = est.estimate(gray);
  ASSERT_TRUE(p.valid);
  EXPECT_NEAR(p.theta_x, 0.0, 0.01);
  EXPECT_NEAR(p.theta_y, 0.0, 0.01);
  EXPECT_LT(p.reproj_err_px, 1.0);
}

TEST(Rate, ConstantPeriod) {
  RateStats r;
  for (int i = 0; i <= 30; ++i) r.tick(i / 30.0);
  auto s = r.summary();
  EXPECT_NEAR(s.hz, 30.0, 1e-6);
  EXPECT_LT(s.jitter_ms, 1e-6);
}
