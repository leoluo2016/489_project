#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <cstdio>
#include <sstream>
#include <optional>
#include <filesystem>
#include <chrono>
#include <iomanip>

using namespace std;
using namespace cv;
using Clock = chrono::high_resolution_clock;

string VIDEO_PATH = "scottie_Trim.mp4";

bool SAVE_TRACKING_VIDEO = true;
string TRACKING_VIDEO_PATH = "two_blob_tracking_debug.mp4";

bool SAVE_WEDGE_VIDEO = true;
string WEDGE_VIDEO_PATH = "two_blob_motion_in_wedge.mp4";

int LAUNCH_SEARCH_START = 240;

string ROI_WINDOW_NAME = "Select Launch ROI";
int ROI_SELECT_MAX_W = 1400;
int ROI_SELECT_MAX_H = 900;
double ROI_SCALE = 1.6;

Scalar HSV_WHITE_LOW(0, 0, 100);
Scalar HSV_WHITE_HIGH(180, 20, 255);

Scalar HSV_YELLOW_LOW(18, 110, 140);
Scalar HSV_YELLOW_HIGH(45, 255, 255);

int COLOR_DIFF_THRESHOLD = 10;

double CONE_TOP_ANGLE_DEG = 38.0;
int BOTTOM_AXIS_ALLOWANCE_PX = 18;
int CONE_STEP_PX = 20;

double MIN_BLOB_AREA = 150;
double MAX_BLOB_AREA = 750;
double MIN_BLOB_RADIUS = 3.0;
double MAX_BLOB_RADIUS = 35.0;

int MIN_PAIR_DX_PX = 15;
int MAX_PAIR_DY_PX = 300;
int PAIR_TO_USE = 4;

double CAPTURE_FPS = 240.0;
double GOLF_BALL_DIAMETER_M = 0.04267;

int DILATE_ITERATIONS = 1;

double total_pixel_stage_ms = 0.0;
double total_dilation_ms = 0.0;
double total_blob_detection_ms = 0.0;
int processed_frames = 0;
int dilation_timed_frames = 0;

struct Blob {
    int cx;
    int cy;
    double area;
    int x;
    int y;
    int w;
    int h;
    double radius;
};

double parse_fraction(const string& s) {
    size_t slash = s.find('/');

    if (slash == string::npos) {
        return stod(s);
    }

    double num = stod(s.substr(0, slash));
    double den = stod(s.substr(slash + 1));

    if (den == 0) {
        return 0.0;
    }

    return num / den;
}

double get_fps_ffprobe(const string& video_path) {
    string command =
        "ffprobe -v error -select_streams v:0 "
        "-show_entries stream=r_frame_rate "
        "-of default=noprint_wrappers=1:nokey=1 \"" + video_path + "\"";

    FILE* pipe = popen(command.c_str(), "r");

    if (!pipe) {
        cerr << "ffprobe failed. Falling back to OpenCV FPS." << endl;
        return 0.0;
    }

    char buffer[128];
    string result;

    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        result += buffer;
    }

    pclose(pipe);

    if (result.empty()) {
        return 0.0;
    }

    result.erase(remove(result.begin(), result.end(), '\n'), result.end());
    result.erase(remove(result.begin(), result.end(), '\r'), result.end());

    return parse_fraction(result);
}

Rect select_roi_scaled(const string& window_name, const Mat& frame, int max_w = 1400, int max_h = 900) {
    int h = frame.rows;
    int w = frame.cols;

    double scale = min({max_w / double(w), max_h / double(h), 1.0});

    Mat disp;

    if (scale < 1.0) {
        resize(frame, disp, Size(int(round(w * scale)), int(round(h * scale))), 0, 0, INTER_AREA);
    } else {
        disp = frame.clone();
    }

    namedWindow(window_name, WINDOW_NORMAL);
    imshow(window_name, disp);

    Rect roi = selectROI(window_name, disp, false, true);
    destroyWindow(window_name);

    int rx = roi.x;
    int ry = roi.y;
    int rw = roi.width;
    int rh = roi.height;

    if (rw <= 0 || rh <= 0) {
        return Rect(0, 0, 0, 0);
    }

    if (scale < 1.0) {
        rx = int(round(rx / scale));
        ry = int(round(ry / scale));
        rw = int(round(rw / scale));
        rh = int(round(rh / scale));
    }

    rx = max(0, min(rx, w - 1));
    ry = max(0, min(ry, h - 1));
    rw = max(1, min(rw, w - rx));
    rh = max(1, min(rh, h - ry));

    return Rect(rx, ry, rw, rh);
}

tuple<string, Scalar, Scalar> choose_ball_color_from_roi(const Mat& frame_bgr, const Rect& roi) {
    Mat roi_bgr = frame_bgr(roi);

    if (roi_bgr.empty()) {
        return {"white", HSV_WHITE_LOW, HSV_WHITE_HIGH};
    }

    Mat roi_hsv;
    cvtColor(roi_bgr, roi_hsv, COLOR_BGR2HSV);

    Mat white_mask;
    Mat yellow_mask;

    inRange(roi_hsv, HSV_WHITE_LOW, HSV_WHITE_HIGH, white_mask);
    inRange(roi_hsv, HSV_YELLOW_LOW, HSV_YELLOW_HIGH, yellow_mask);

    int white_count = countNonZero(white_mask);
    int yellow_count = countNonZero(yellow_mask);

    cout << "ROI color check: white_pixels=" << white_count
         << ", yellow_pixels=" << yellow_count << endl;

    if (yellow_count > white_count) {
        cout << "Using yellow HSV threshold." << endl;
        return {"yellow", HSV_YELLOW_LOW, HSV_YELLOW_HIGH};
    }

    cout << "Using white HSV threshold." << endl;
    return {"white", HSV_WHITE_LOW, HSV_WHITE_HIGH};
}

Rect expand_rect(int x, int y, int w, int h, double scale, int frame_w, int frame_h) {
    double cx = x + w / 2.0;
    double cy = y + h / 2.0;

    int new_w = max(2, int(round(w * scale)));
    int new_h = max(2, int(round(h * scale)));

    int x0 = int(round(cx - new_w / 2.0));
    int y0 = int(round(cy - new_h / 2.0));
    int x1 = x0 + new_w;
    int y1 = y0 + new_h;

    x0 = max(0, x0);
    y0 = max(0, y0);
    x1 = min(frame_w, x1);
    y1 = min(frame_h, y1);

    return Rect(x0, y0, x1 - x0, y1 - y0);
}

Mat build_wedge_mask(Size shape, Point origin) {
    int h = shape.height;
    int w = shape.width;

    Mat mask = Mat::zeros(h, w, CV_8UC1);

    int ox = origin.x;
    int oy = origin.y;

    int top_end_y = int(max(0.0, oy - (w - 1 - ox) * tan(CONE_TOP_ANGLE_DEG * CV_PI / 180.0)));

    vector<Point> pts = {
        Point(ox, oy),
        Point(w - 1, min(h - 1, oy + BOTTOM_AXIS_ALLOWANCE_PX)),
        Point(w - 1, top_end_y)
    };

    fillPoly(mask, vector<vector<Point>>{pts}, Scalar(255));

    return mask;
}

bool point_in_rect(int px, int py, const Rect& rect) {
    return px >= rect.x && px <= rect.x + rect.width &&
           py >= rect.y && py <= rect.y + rect.height;
}

bool point_in_wedge(int px, int py, Point origin) {
    int x0 = origin.x;
    int y0 = origin.y;

    int dx = px - x0;

    if (dx < 0) {
        return false;
    }

    double top_y = y0 - dx * tan(CONE_TOP_ANGLE_DEG * CV_PI / 180.0);
    double bottom_y = y0 + BOTTOM_AXIS_ALLOWANCE_PX;

    return top_y <= py && py <= bottom_y;
}

vector<Blob> get_blobs_from_binary(Mat binary_img, const Rect& launch_rect, Point launch_origin) {
    if (binary_img.type() != CV_8UC1) {
        binary_img.convertTo(binary_img, CV_8UC1);
    }

    threshold(binary_img, binary_img, 1, 255, THRESH_BINARY);

    vector<vector<Point>> contours;
    findContours(binary_img, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    vector<Blob> blobs;

    for (const auto& c : contours) {
        double area = contourArea(c);

        if (area < MIN_BLOB_AREA || area > MAX_BLOB_AREA) {
            continue;
        }

        Moments M = moments(c);

        if (abs(M.m00) < 1e-9) {
            continue;
        }

        int cx = int(round(M.m10 / M.m00));
        int cy = int(round(M.m01 / M.m00));

        Rect r = boundingRect(c);
        double radius_est = 0.5 * max(r.width, r.height);

        if (radius_est < MIN_BLOB_RADIUS || radius_est > MAX_BLOB_RADIUS) {
            continue;
        }

        if (!point_in_wedge(cx, cy, launch_origin)) {
            cout << "Blob at (" << cx << "," << cy << ") rejected for being outside launch wedge." << endl;
            continue;
        }

        blobs.push_back({
            cx,
            cy,
            area,
            r.x,
            r.y,
            r.width,
            r.height,
            radius_est
        });
    }

    return blobs;
}

optional<pair<Blob, Blob>> find_best_two_blob_pair(vector<Blob> blobs) {
    if (blobs.size() < 2) {
        return nullopt;
    }

    sort(blobs.begin(), blobs.end(), [](const Blob& a, const Blob& b) {
        return a.cx < b.cx;
    });

    Blob left = blobs[blobs.size() - 2];
    Blob right = blobs[blobs.size() - 1];

    int dx = right.cx - left.cx;
    int dy = right.cy - left.cy;

    if (dx < MIN_PAIR_DX_PX) {
        return nullopt;
    }

    if (abs(dy) > MAX_PAIR_DY_PX) {
        return nullopt;
    }

    return pair<Blob, Blob>(left, right);
}

tuple<double, double, double, double> compute_launch_from_pair(const Blob& left_blob, const Blob& right_blob, double fps) {
    double dt = 1.0 / fps;

    double dx = right_blob.cx - left_blob.cx;
    double dy_up = -(right_blob.cy - left_blob.cy);

    double vx_pix = dx / dt;
    double vy_pix = dy_up / dt;
    double speed_pix = hypot(vx_pix, vy_pix);
    double launch_angle_deg = atan2(vy_pix, vx_pix) * 180.0 / CV_PI;

    return {vx_pix, vy_pix, speed_pix, launch_angle_deg};
}

struct RealSpeed {
    double displacement_px;
    double ball_diameter_px;
    double meters_per_pixel;
    double vx_mps;
    double vy_mps;
    double speed_mps;
    double speed_mph;
};

optional<RealSpeed> compute_real_world_speed_from_pair(const Blob& left_blob, const Blob& right_blob, double capture_fps) {
    double dx_px = right_blob.cx - left_blob.cx;
    double dy_px = right_blob.cy - left_blob.cy;
    double displacement_px = hypot(dx_px, dy_px);

    double avg_radius_px = 0.5 * (left_blob.radius + right_blob.radius);
    double ball_diameter_px = 2.0 * avg_radius_px;

    if (ball_diameter_px <= 0) {
        return nullopt;
    }

    double meters_per_pixel = GOLF_BALL_DIAMETER_M / ball_diameter_px;

    double speed_mps = displacement_px * capture_fps * meters_per_pixel;
    double speed_mph = speed_mps * 2.2369362920544;

    double vx_mps = dx_px * capture_fps * meters_per_pixel;
    double vy_mps = -dy_px * capture_fps * meters_per_pixel;

    return RealSpeed{
        displacement_px,
        ball_diameter_px,
        meters_per_pixel,
        vx_mps,
        vy_mps,
        speed_mps,
        speed_mph
    };
}

void draw_wedge_guides(Mat& frame, Point origin) {
    int h = frame.rows;
    int w = frame.cols;

    int ox = origin.x;
    int oy = origin.y;

    for (int x = ox; x < w; x += CONE_STEP_PX) {
        int dx = x - ox;
        int top = int(max(0.0, oy - dx * tan(CONE_TOP_ANGLE_DEG * CV_PI / 180.0)));
        int bottom = int(min(h - 1, oy + BOTTOM_AXIS_ALLOWANCE_PX));
        line(frame, Point(x, top), Point(x, bottom), Scalar(255, 0, 0), 1);
    }

    line(frame, Point(ox, oy), Point(w - 1, oy), Scalar(255, 255, 0), 2);

    int top_end_y = int(max(0.0, oy - (w - 1 - ox) * tan(CONE_TOP_ANGLE_DEG * CV_PI / 180.0)));
    line(frame, Point(ox, oy), Point(w - 1, top_end_y), Scalar(255, 255, 0), 2);
}

int main() {
    filesystem::path SCRIPT_DIR = filesystem::current_path();

    VIDEO_PATH = (SCRIPT_DIR / VIDEO_PATH).string();
    TRACKING_VIDEO_PATH = (SCRIPT_DIR / TRACKING_VIDEO_PATH).string();
    WEDGE_VIDEO_PATH = (SCRIPT_DIR / WEDGE_VIDEO_PATH).string();

    VideoCapture cap(VIDEO_PATH);

    if (!cap.isOpened()) {
        throw runtime_error("Could not open video: " + VIDEO_PATH);
    }

    double fps = get_fps_ffprobe(VIDEO_PATH);

    if (fps <= 0) {
        fps = cap.get(CAP_PROP_FPS);
    }

    CAPTURE_FPS = 240.0;

    cout << "Using FPS: " << fps << endl;

    Mat first_frame;
    bool ret = cap.read(first_frame);

    if (!ret) {
        throw runtime_error("Could not read first frame");
    }

    int frame_h = first_frame.rows;
    int frame_w = first_frame.cols;

    cout << "Select the launch ball/tee ROI." << endl;

    Rect roi = select_roi_scaled(
        ROI_WINDOW_NAME,
        first_frame,
        ROI_SELECT_MAX_W,
        ROI_SELECT_MAX_H
    );

    int rx = roi.x;
    int ry = roi.y;
    int rw = roi.width;
    int rh = roi.height;

    if (rw <= 0 || rh <= 0) {
        throw runtime_error("No ROI selected.");
    }

    auto [ball_color_name, hsv_low, hsv_high] = choose_ball_color_from_roi(first_frame, roi);

    Rect launch_rect = expand_rect(rx, ry, rw, rh, ROI_SCALE, frame_w, frame_h);
    Point launch_origin(rx + rw / 2, ry + rh / 2);

    cout << "Launch ROI: x=" << rx << ", y=" << ry << ", w=" << rw << ", h=" << rh << endl;
    cout << "Expanded launch rect: x=" << launch_rect.x << ", y=" << launch_rect.y
         << ", w=" << launch_rect.width << ", h=" << launch_rect.height << endl;
    cout << "Wedge origin: " << launch_origin << endl;
    cout << "Press p or Space while one of the OpenCV video windows is focused to pause/resume. Press q or Esc to quit." << endl;

    cap.set(CAP_PROP_POS_FRAMES, 0);

    VideoWriter tracking_writer;

    if (SAVE_TRACKING_VIDEO) {
        int fourcc = VideoWriter::fourcc('m', 'p', '4', 'v');
        tracking_writer.open(TRACKING_VIDEO_PATH, fourcc, fps, Size(frame_w, frame_h), true);
        cout << "Saving tracking mp4 to: " << TRACKING_VIDEO_PATH << endl;
    }

    VideoWriter wedge_writer;

    if (SAVE_WEDGE_VIDEO) {
        int fourcc = VideoWriter::fourcc('m', 'p', '4', 'v');
        wedge_writer.open(WEDGE_VIDEO_PATH, fourcc, fps, Size(frame_w, frame_h), false);
        cout << "Saving wedge mp4 to: " << WEDGE_VIDEO_PATH << endl;
    }

    Mat kernel = Mat::ones(2, 2, CV_8U);

    Mat prev_color_mask;
    bool paused = false;
    Mat last_display_frame;

    optional<tuple<double, double, double, double>> result;
    Mat final_frame;
    vector<Blob> final_blobs;
    optional<pair<Blob, Blob>> final_pair;
    optional<int> final_frame_idx;
    int valid_pair_count = 0;
    bool detection_locked = false;

    namedWindow("Color-Mask Difference", WINDOW_NORMAL);
    namedWindow("Motion In Wedge", WINDOW_NORMAL);
    namedWindow("Tracking", WINDOW_NORMAL);

    int frame_idx = -1;

    double total_pixel_stage_ms = 0.0;
    double total_blob_detection_ms = 0.0;
    int processed_frames = 0;

    auto total_start = Clock::now();

    while (true) {
        Mat frame;
        bool ret = cap.read(frame);

        if (!ret) {
            break;
        }

        frame_idx++;

        Mat display = frame.clone();

        if (paused) {
            Mat pause_display = !last_display_frame.empty() ? last_display_frame.clone() : display.clone();

            putText(pause_display, "PAUSED", Point(20, 40), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 255, 255), 2);
            imshow("Tracking", pause_display);

            int key = waitKey(30) & 0xFF;

            if (key == 'q' || key == 27) {
                break;
            }

            if (key == 'p' || key == 32) {
                paused = false;
            }

            continue;
        }

        auto pixel_start = Clock::now();

        Mat hsv;
        cvtColor(frame, hsv, COLOR_BGR2HSV);

        Mat color_mask;
        inRange(hsv, hsv_low, hsv_high, color_mask);

        if (prev_color_mask.empty()) {
            prev_color_mask = color_mask.clone();

            Mat zero = Mat::zeros(color_mask.size(), color_mask.type());
            last_display_frame = display.clone();

            imshow("Color-Mask Difference", zero);
            imshow("Motion In Wedge", zero);
            imshow("Tracking", display);

            int key = waitKey(30) & 0xFF;

            if (key == 'q' || key == 27) {
                break;
            }

            if (key == 'p' || key == 32) {
                paused = true;
            }

            continue;
        }

        Mat color_diff;
        absdiff(color_mask, prev_color_mask, color_diff);

        Mat motion_seed;
        threshold(color_diff, motion_seed, COLOR_DIFF_THRESHOLD, 255, THRESH_BINARY);

        Mat wedge_mask = build_wedge_mask(motion_seed.size(), launch_origin);

        Mat motion_in_wedge;
        bitwise_and(motion_seed, wedge_mask, motion_in_wedge);

        Mat final_color_mask;
        Mat final_motion_seed;
        Mat final_motion_in_wedge;

        if (DILATE_ITERATIONS > 0) {
            auto dilation_start = Clock::now();

            dilate(
                motion_in_wedge,
                motion_in_wedge,
                kernel,
                Point(-1, -1),
                DILATE_ITERATIONS
            );

            auto dilation_end = Clock::now();

            total_dilation_ms += chrono::duration<double, milli>(
                dilation_end - dilation_start
            ).count();

            dilation_timed_frames++;
        }

        auto pixel_end = Clock::now();
        total_pixel_stage_ms += chrono::duration<double, milli>(pixel_end - pixel_start).count();

        if (wedge_writer.isOpened()) {
            wedge_writer.write(motion_in_wedge);
        }

        auto blob_start = Clock::now();

        vector<Blob> blobs;

        if (frame_idx >= LAUNCH_SEARCH_START) {
            blobs = get_blobs_from_binary(motion_in_wedge, launch_rect, launch_origin);
        }

        optional<pair<Blob, Blob>> pair_result = find_best_two_blob_pair(blobs);

        auto blob_end = Clock::now();
        total_blob_detection_ms += chrono::duration<double, milli>(blob_end - blob_start).count();

        rectangle(display, launch_rect, Scalar(255, 255, 0), 2);
        circle(display, launch_origin, 4, Scalar(255, 0, 255), -1);
        draw_wedge_guides(display, launch_origin);

        for (const Blob& b : blobs) {
            rectangle(display, Rect(b.x, b.y, b.w, b.h), Scalar(0, 255, 0), 1);
            circle(display, Point(b.cx, b.cy), 4, Scalar(0, 255, 0), -1);

            string text = "A=" + to_string(int(b.area)) + " R=" + to_string(b.radius).substr(0, 4);

            putText(
                display,
                text,
                Point(b.cx + 6, b.cy - 6),
                FONT_HERSHEY_SIMPLEX,
                0.45,
                Scalar(0, 255, 0),
                1,
                LINE_AA
            );
        }

        if (pair_result.has_value() && !detection_locked) {
            valid_pair_count++;

            Blob left_blob = pair_result->first;
            Blob right_blob = pair_result->second;

            cout << "Valid pair " << valid_pair_count << " at frame " << frame_idx
                 << ": L=(" << left_blob.cx << "," << left_blob.cy << "), area=" << left_blob.area
                 << ", radius=" << left_blob.radius
                 << "; R=(" << right_blob.cx << "," << right_blob.cy << "), area=" << right_blob.area
                 << ", radius=" << right_blob.radius << endl;

            circle(display, Point(left_blob.cx, left_blob.cy), 10, Scalar(255, 0, 0), 1);
            circle(display, Point(right_blob.cx, right_blob.cy), 10, Scalar(255, 0, 0), 1);
            line(display, Point(left_blob.cx, left_blob.cy), Point(right_blob.cx, right_blob.cy), Scalar(255, 0, 0), 1);

            if (valid_pair_count >= PAIR_TO_USE) {
                result = compute_launch_from_pair(left_blob, right_blob, fps);

                auto [vx_pix, vy_pix, speed_pix, launch_angle_deg] = result.value();

                optional<RealSpeed> real_speed = compute_real_world_speed_from_pair(left_blob, right_blob, CAPTURE_FPS);

                cout << "\nTwo-blob launch detected using valid pair " << valid_pair_count
                     << " at frame " << frame_idx << endl;

                cout << "Left/previous blob: center=(" << left_blob.cx << "," << left_blob.cy
                     << "), area=" << left_blob.area << ", radius=" << left_blob.radius << endl;

                cout << "Right/current blob: center=(" << right_blob.cx << "," << right_blob.cy
                     << "), area=" << right_blob.area << ", radius=" << right_blob.radius << endl;

                cout << "vx_pix = " << vx_pix << " px/s" << endl;
                cout << "vy_pix = " << vy_pix << " px/s" << endl;
                cout << "speed_pix = " << speed_pix << " px/s" << endl;
                cout << "launch_angle_deg = " << launch_angle_deg << endl;

                if (real_speed.has_value()) {
                    cout << "\nApproximate real-world speed estimate:" << endl;
                    cout << "  CAPTURE_FPS = " << CAPTURE_FPS << endl;
                    cout << "  ball_diameter_px = " << real_speed->ball_diameter_px << " px" << endl;
                    cout << "  meters_per_pixel = " << real_speed->meters_per_pixel << " m/px" << endl;
                    cout << "  vx_mps = " << real_speed->vx_mps << " m/s" << endl;
                    cout << "  vy_mps = " << real_speed->vy_mps << " m/s" << endl;
                    cout << "  speed_mps = " << real_speed->speed_mps << " m/s" << endl;
                    cout << "  speed_mph = " << real_speed->speed_mph << " mph" << endl;
                }

                circle(display, Point(left_blob.cx, left_blob.cy), 12, Scalar(0, 255, 255), 2);
                circle(display, Point(right_blob.cx, right_blob.cy), 12, Scalar(0, 0, 255), 2);
                line(display, Point(left_blob.cx, left_blob.cy), Point(right_blob.cx, right_blob.cy), Scalar(0, 255, 255), 2);

                putText(display, "P1 prev", Point(left_blob.cx + 10, left_blob.cy - 10), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 255), 2);
                putText(display, "P2 curr", Point(right_blob.cx + 10, right_blob.cy - 10), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 0, 255), 2);

                putText(display, "speed=" + to_string(speed_pix).substr(0, 8) + " px/s", Point(20, 35), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 255, 255), 2);
                putText(display, "angle=" + to_string(launch_angle_deg).substr(0, 6) + " deg", Point(20, 70), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 255, 255), 2);

                if (real_speed.has_value()) {
                    putText(display, "approx=" + to_string(real_speed->speed_mph).substr(0, 6) + " mph @ " + to_string(int(CAPTURE_FPS)) + " fps",
                            Point(20, 105), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 255, 255), 2);
                }

                last_display_frame = display.clone();

                final_frame = display.clone();
                final_color_mask = color_mask.clone();
                final_motion_seed = motion_seed.clone();
                final_motion_in_wedge = motion_in_wedge.clone();

                imwrite("final_two_blob_launch_frame.png", final_frame);
                imwrite("final_color_mask.png", final_color_mask);
                imwrite("final_frame_difference.png", final_motion_seed);
                imwrite("final_motion_in_wedge.png", final_motion_in_wedge);

                final_blobs = blobs;
                final_pair = pair_result;
                final_frame_idx = frame_idx;
                detection_locked = true;

                imshow("Color-Mask Difference", motion_seed);
                imshow("Motion In Wedge", motion_in_wedge);
                imshow("Tracking", display);
                waitKey(1);
            }
        }

        if (detection_locked && result.has_value() && final_pair.has_value()) {
            Blob left_blob = final_pair->first;
            Blob right_blob = final_pair->second;

            auto [vx_pix, vy_pix, speed_pix, launch_angle_deg] = result.value();
            optional<RealSpeed> real_speed = compute_real_world_speed_from_pair(left_blob, right_blob, CAPTURE_FPS);

            circle(display, Point(left_blob.cx, left_blob.cy), 12, Scalar(0, 255, 255), 2);
            circle(display, Point(right_blob.cx, right_blob.cy), 12, Scalar(0, 0, 255), 2);
            line(display, Point(left_blob.cx, left_blob.cy), Point(right_blob.cx, right_blob.cy), Scalar(0, 255, 255), 2);

            putText(display, "P1 prev", Point(left_blob.cx + 10, left_blob.cy - 10), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 255), 2);
            putText(display, "P2 curr", Point(right_blob.cx + 10, right_blob.cy - 10), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 0, 255), 2);

            putText(display, "speed=" + to_string(speed_pix).substr(0, 8) + " px/s", Point(20, 35), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 255, 255), 2);
            putText(display, "angle=" + to_string(launch_angle_deg).substr(0, 6) + " deg", Point(20, 70), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 255, 255), 2);

            if (real_speed.has_value()) {
                putText(display, "approx=" + to_string(real_speed->speed_mph).substr(0, 6) + " mph @ " + to_string(int(CAPTURE_FPS)) + " fps",
                        Point(20, 105), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 255, 255), 2);
            }
        }

        if (tracking_writer.isOpened()) {
            tracking_writer.write(display);
        }

        last_display_frame = display.clone();

        imshow("Color-Mask Difference", motion_seed);
        imshow("Motion In Wedge", motion_in_wedge);
        imshow("Tracking", display);

        prev_color_mask = color_mask.clone();

        processed_frames++;

        int key = waitKey(1) & 0xFF;

        if (key == 'q' || key == 27) {
            break;
        }

        if (key == 'p' || key == 32) {
            paused = true;
        }
    }

    auto total_end = Clock::now();
    double total_runtime_ms = chrono::duration<double, milli>(total_end - total_start).count();

    cout << fixed << setprecision(3);
    cout << "\n================ Timing Results ================\n";
    cout << "Frames processed: " << processed_frames << "\n";
    cout << "Total runtime: " << total_runtime_ms / 1000.0 << " s\n";
    cout << "Average processing time per frame: "
         << (processed_frames > 0 ? total_runtime_ms / processed_frames : 0.0)
         << " ms/frame\n";
    cout << "Average pixel stage time per frame: "
         << (processed_frames > 0 ? total_pixel_stage_ms / processed_frames : 0.0)
         << " ms/frame\n";
    cout << "Average blob detection time per frame: "
         << (processed_frames > 0 ? total_blob_detection_ms / processed_frames : 0.0)
         << " ms/frame\n";
    cout << "Pixel stage total time: " << total_pixel_stage_ms / 1000.0 << " s\n";
    cout << "Blob detection total time: " << total_blob_detection_ms / 1000.0 << " s\n";
    cout << "================================================\n";
    cout << "Average CPU dilation time per frame: "
         << (dilation_timed_frames > 0 ? total_dilation_ms / dilation_timed_frames : 0.0)
         << " ms/frame\n";

    cout << "CPU dilation total time: "
         << total_dilation_ms / 1000.0
         << " s\n";

    cap.release();

    if (tracking_writer.isOpened()) {
        tracking_writer.release();
    }

    if (wedge_writer.isOpened()) {
        wedge_writer.release();
    }

    if (!final_frame.empty()) {
        namedWindow("Final Two-Blob Launch Frame", WINDOW_NORMAL);

        cout << "\nFinal frame displayed." << endl;
        cout << "Press q or Esc to close." << endl;

        while (true) {
            imshow("Final Two-Blob Launch Frame", final_frame);

            int key = waitKey(30) & 0xFF;

            if (key == 'q' || key == 27) {
                break;
            }
        }
    } else {
        cout << "\nNo two-blob launch pair detected using contour moments." << endl;
        cout << "If the blobs are visible in Motion In Wedge, loosen MIN_BLOB_AREA, MIN_BLOB_RADIUS, or MIN_PAIR_DX_PX. This version uses contour moments." << endl;
    }

    destroyAllWindows();

    return 0;
}