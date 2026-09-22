#include <SFML/Graphics.hpp>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err__ = (call);                                               \
        if (err__ != cudaSuccess) {                                               \
            std::cerr << "CUDA error: " << cudaGetErrorString(err__)             \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;     \
            std::exit(EXIT_FAILURE);                                              \
        }                                                                         \
    } while (0)

namespace sim {

constexpr double kPi = 3.14159265358979323846;
constexpr double kGravityFtPerSec2 = 32.174;
constexpr double kAirDensitySlugPerFt3 = 0.0023769;
constexpr double kBallRadiusFt = 1.68 / 24.0;
constexpr double kBallAreaFt2 = kPi * kBallRadiusFt * kBallRadiusFt;
constexpr double kBallMassSlug = 0.1012 / 32.174;
constexpr double kSecondsPerHour = 3600.0;
constexpr double kFeetPerMile = 5280.0;
constexpr int kMaxSamples = 2500;

struct Inputs {
    double ball_speed_mph = 128.0;
    double launch_angle_deg = 19.2;
    double spin_rate_rpm = 5400.0;
};

struct Sample {
    double time_s = 0.0;
    double x_ft = 0.0;
    double y_ft = 0.0;
    double vx_ft_s = 0.0;
    double vy_ft_s = 0.0;
};

struct Result {
    std::vector<Sample> trajectory;
    double carry_distance_ft = 0.0;
    double carry_distance_yd = 0.0;
    double flight_time_s = 0.0;
};

struct SweepEntry {
    Inputs inputs;
    Result result;
    bool is_reference = false;
};

struct GpuResultHeader {
    double carry_ft = 0.0;
    double carry_yd = 0.0;
    double flight_time_s = 0.0;
    int sample_count = 0;
};

__host__ __device__ double clamp_double(double x, double lo, double hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

__host__ __device__ double max_double(double a, double b) {
    return a > b ? a : b;
}

__host__ __device__ double abs_double(double x) {
    return x < 0.0 ? -x : x;
}

__host__ __device__ double mph_to_ft_per_s(double mph) {
    return mph * kFeetPerMile / kSecondsPerHour;
}

__host__ __device__ double deg_to_rad(double deg) {
    return deg * kPi / 180.0;
}

__host__ __device__ double speed_ft_per_s(const double s[4]) {
    return sqrt(s[1] * s[1] + s[3] * s[3]);
}

// These are intentionally the same coefficients as the CPU tracer.cpp baseline.
__host__ __device__ double drag_coefficient(double speed_mph) {
    const double cd_high = 0.075;
    const double cd_low = 0.34;
    const double transition_mph = 130.0;
    const double width_mph = 15.0;
    return cd_high + (cd_low - cd_high) / (1.0 + exp((speed_mph - transition_mph) / width_mph));
}

__host__ __device__ double spin_decay_tau(double speed_mph) {
    return 5.0 + 7.0 / (1.0 + exp(-(speed_mph - 130.0) / 15.0));
}

// These are intentionally the same coefficients as the CPU tracer.cpp baseline.
__host__ __device__ double lift_coefficient(double spin_ratio) {
    const double slope = 1.4;
    const double max_lift = 0.50;
    const double value = slope * spin_ratio;
    return clamp_double(value, 0.0, max_lift);
}

__host__ __device__ void derivative(double time_s, const double state[4], double spin_rate_rpm, double out[4]) {
    const double vx = state[1];
    const double vy = state[3];
    const double speed = speed_ft_per_s(state);
    const double safe_speed = max_double(speed, 1e-6);
    const double speed_mph = speed * kSecondsPerHour / kFeetPerMile;
    const double cd = drag_coefficient(speed_mph);

    const double omega0 = spin_rate_rpm * 2.0 * kPi / 60.0;
    const double tau = spin_decay_tau(speed_mph);
    const double omega = omega0 * exp(-time_s / tau);
    const double spin_ratio = omega * kBallRadiusFt / safe_speed;
    const double cl = lift_coefficient(spin_ratio);

    const double drag_force = 0.5 * kAirDensitySlugPerFt3 * kBallAreaFt2 * cd * speed * speed;
    const double lift_force = 0.5 * kAirDensitySlugPerFt3 * kBallAreaFt2 * cl * speed * speed;

    const double drag_x = -drag_force * vx / safe_speed;
    const double drag_y = -drag_force * vy / safe_speed;
    const double lift_x = -lift_force * vy / safe_speed;
    const double lift_y = lift_force * vx / safe_speed;

    out[0] = vx;
    out[1] = (drag_x + lift_x) / kBallMassSlug;
    out[2] = vy;
    out[3] = -kGravityFtPerSec2 + (drag_y + lift_y) / kBallMassSlug;
}

__host__ __device__ void add_scaled(const double a[4], const double b[4], double scale, double out[4]) {
    for (int i = 0; i < 4; ++i) out[i] = a[i] + scale * b[i];
}

__host__ __device__ double max_abs_diff(const double a[4], const double b[4]) {
    double max_value = 0.0;
    for (int i = 0; i < 4; ++i) max_value = max_double(max_value, abs_double(a[i] - b[i]));
    return max_value;
}

__host__ __device__ void rk45_step(double time_s,
                                   const double state[4],
                                   double step_s,
                                   double spin_rate_rpm,
                                   double next_state[4],
                                   double embedded_state[4]) {
    double k1[4], k2[4], k3[4], k4[4], k5[4], k6[4], k7[4];
    double stage[4];

    derivative(time_s, state, spin_rate_rpm, k1);

    add_scaled(state, k1, step_s * (1.0 / 5.0), stage);
    derivative(time_s + step_s * (1.0 / 5.0), stage, spin_rate_rpm, k2);

    for (int i = 0; i < 4; ++i) {
        stage[i] = state[i] + step_s * ((3.0 / 40.0) * k1[i] + (9.0 / 40.0) * k2[i]);
    }
    derivative(time_s + step_s * (3.0 / 10.0), stage, spin_rate_rpm, k3);

    for (int i = 0; i < 4; ++i) {
        stage[i] = state[i] + step_s * ((44.0 / 45.0) * k1[i] + (-56.0 / 15.0) * k2[i] + (32.0 / 9.0) * k3[i]);
    }
    derivative(time_s + step_s * (4.0 / 5.0), stage, spin_rate_rpm, k4);

    for (int i = 0; i < 4; ++i) {
        stage[i] = state[i] + step_s * ((19372.0 / 6561.0) * k1[i] + (-25360.0 / 2187.0) * k2[i] +
                                        (64448.0 / 6561.0) * k3[i] + (-212.0 / 729.0) * k4[i]);
    }
    derivative(time_s + step_s * (8.0 / 9.0), stage, spin_rate_rpm, k5);

    for (int i = 0; i < 4; ++i) {
        stage[i] = state[i] + step_s * ((9017.0 / 3168.0) * k1[i] + (-355.0 / 33.0) * k2[i] +
                                        (46732.0 / 5247.0) * k3[i] + (49.0 / 176.0) * k4[i] +
                                        (-5103.0 / 18656.0) * k5[i]);
    }
    derivative(time_s + step_s, stage, spin_rate_rpm, k6);

    for (int i = 0; i < 4; ++i) {
        next_state[i] = state[i] + step_s * ((35.0 / 384.0) * k1[i] +
                                             (500.0 / 1113.0) * k3[i] +
                                             (125.0 / 192.0) * k4[i] +
                                             (-2187.0 / 6784.0) * k5[i] +
                                             (11.0 / 84.0) * k6[i]);
    }

    derivative(time_s + step_s, next_state, spin_rate_rpm, k7);

    for (int i = 0; i < 4; ++i) {
        embedded_state[i] = state[i] + step_s * ((5179.0 / 57600.0) * k1[i] +
                                                 (7571.0 / 16695.0) * k3[i] +
                                                 (393.0 / 640.0) * k4[i] +
                                                 (-92097.0 / 339200.0) * k5[i] +
                                                 (187.0 / 2100.0) * k6[i] +
                                                 (1.0 / 40.0) * k7[i]);
    }
}

__host__ __device__ Sample make_sample(double time_s, const double s[4]) {
    Sample sample;
    sample.time_s = time_s;
    sample.x_ft = s[0];
    sample.vx_ft_s = s[1];
    sample.y_ft = s[2];
    sample.vy_ft_s = s[3];
    return sample;
}

__host__ __device__ Sample interpolate_ground_hit(const Sample& above, const Sample& below) {
    const double y0 = above.y_ft;
    const double y1 = below.y_ft;
    double fraction = 0.0;
    if (abs_double(y1 - y0) > 1e-12) fraction = (0.0 - y0) / (y1 - y0);
    fraction = clamp_double(fraction, 0.0, 1.0);

    Sample hit;
    hit.time_s = above.time_s + fraction * (below.time_s - above.time_s);
    hit.x_ft = above.x_ft + fraction * (below.x_ft - above.x_ft);
    hit.y_ft = 0.0;
    hit.vx_ft_s = above.vx_ft_s + fraction * (below.vx_ft_s - above.vx_ft_s);
    hit.vy_ft_s = above.vy_ft_s + fraction * (below.vy_ft_s - above.vy_ft_s);
    return hit;
}

__device__ void simulate_one_cuda(const Inputs& inputs, Sample* out, GpuResultHeader& header) {
    const double speed0 = mph_to_ft_per_s(inputs.ball_speed_mph);
    const double theta = deg_to_rad(inputs.launch_angle_deg);

    double state[4] = {0.0, speed0 * cos(theta), 0.01, speed0 * sin(theta)};
    double time_s = 0.0;
    double step_s = 0.01;

    constexpr double end_time_s = 20.0;
    constexpr double abs_tol = 1e-10;
    constexpr double rel_tol = 1e-8;
    constexpr double min_step_s = 1e-6;
    constexpr double max_step_s = 0.05;

    int sample_count = 0;
    Sample sample = make_sample(time_s, state);
    out[sample_count++] = sample;

    while (time_s < end_time_s && sample_count < kMaxSamples - 1) {
        if (time_s + step_s > end_time_s) step_s = end_time_s - time_s;

        double next_state[4];
        double embedded_state[4];
        rk45_step(time_s, state, step_s, inputs.spin_rate_rpm, next_state, embedded_state);

        double scale = 0.0;
        for (int i = 0; i < 4; ++i) scale = max_double(scale, abs_double(next_state[i]));
        const double tolerance = abs_tol + rel_tol * max_double(1.0, scale);
        const double error = max_abs_diff(next_state, embedded_state);

        if (error <= tolerance) {
            Sample next_sample = make_sample(time_s + step_s, next_state);

            if (next_sample.y_ft <= 0.0) {
                Sample hit = interpolate_ground_hit(sample, next_sample);
                out[sample_count++] = hit;
                header.carry_ft = hit.x_ft;
                header.carry_yd = hit.x_ft / 3.0;
                header.flight_time_s = hit.time_s;
                header.sample_count = sample_count;
                return;
            }

            out[sample_count++] = next_sample;
            for (int i = 0; i < 4; ++i) state[i] = next_state[i];
            sample = next_sample;
            time_s = next_sample.time_s;

            double factor = 1.5;
            if (error > 0.0) factor = 0.9 * pow(tolerance / error, 0.2);
            step_s = clamp_double(step_s * factor, min_step_s, max_step_s);
        } else {
            const double factor = 0.9 * pow(tolerance / max_double(error, 1e-16), 0.25);
            step_s = clamp_double(step_s * max_double(0.1, factor), min_step_s, max_step_s);
        }
    }

    Sample last = out[sample_count - 1];
    header.carry_ft = last.x_ft;
    header.carry_yd = last.x_ft / 3.0;
    header.flight_time_s = last.time_s;
    header.sample_count = sample_count;
}

__global__ void simulate_kernel(const Inputs* inputs, int n, Sample* samples, GpuResultHeader* headers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    simulate_one_cuda(inputs[idx], samples + idx * kMaxSamples, headers[idx]);
}

std::vector<SweepEntry> build_spin_sweep_inputs(const Inputs& reference_inputs) {
    std::vector<SweepEntry> sweep;
    for (int offset = -1000; offset <= 1000; offset += 250) {
        Inputs inputs = reference_inputs;
        inputs.spin_rate_rpm = std::max(0.0, reference_inputs.spin_rate_rpm + static_cast<double>(offset));
        SweepEntry entry;
        entry.inputs = inputs;
        entry.is_reference = std::abs(inputs.spin_rate_rpm - reference_inputs.spin_rate_rpm) < 1e-9;
        sweep.push_back(entry);
    }
    return sweep;
}

std::vector<SweepEntry> simulate_sweep_cuda(std::vector<SweepEntry> sweep, const Inputs& reference_inputs, float& gpu_ms, int& trajectories_computed) {
    // Match the CPU timing behavior: compute the single reference shot plus the sweep.
    std::vector<Inputs> h_inputs;
    h_inputs.reserve(sweep.size() + 1);
    h_inputs.push_back(reference_inputs);
    for (const auto& entry : sweep) h_inputs.push_back(entry.inputs);

    const int n = static_cast<int>(h_inputs.size());
    trajectories_computed = n;

    Inputs* d_inputs = nullptr;
    Sample* d_samples = nullptr;
    GpuResultHeader* d_headers = nullptr;

    std::vector<Sample> h_samples(static_cast<std::size_t>(n) * kMaxSamples);
    std::vector<GpuResultHeader> h_headers(n);

    CUDA_CHECK(cudaMalloc(&d_inputs, n * sizeof(Inputs)));
    CUDA_CHECK(cudaMalloc(&d_samples, static_cast<std::size_t>(n) * kMaxSamples * sizeof(Sample)));
    CUDA_CHECK(cudaMalloc(&d_headers, n * sizeof(GpuResultHeader)));

    CUDA_CHECK(cudaMemcpy(d_inputs, h_inputs.data(), n * sizeof(Inputs), cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    dim3 block(128);
    dim3 grid((n + block.x - 1) / block.x);

    // This timing matches CUDA computation time: kernel only, not SFML animation.
    CUDA_CHECK(cudaEventRecord(start));
    simulate_kernel<<<grid, block>>>(d_inputs, n, d_samples, d_headers);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, start, stop));

    CUDA_CHECK(cudaMemcpy(h_samples.data(), d_samples, static_cast<std::size_t>(n) * kMaxSamples * sizeof(Sample), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_headers.data(), d_headers, n * sizeof(GpuResultHeader), cudaMemcpyDeviceToHost));

    // Index 0 is the reference shot for timing; sweep rows start at index 1.
    for (int i = 0; i < static_cast<int>(sweep.size()); ++i) {
        const int src = i + 1;
        Result r;
        r.carry_distance_ft = h_headers[src].carry_ft;
        r.carry_distance_yd = h_headers[src].carry_yd;
        r.flight_time_s = h_headers[src].flight_time_s;
        int count = std::max(0, std::min(h_headers[src].sample_count, kMaxSamples));
        r.trajectory.assign(h_samples.begin() + static_cast<std::size_t>(src) * kMaxSamples,
                            h_samples.begin() + static_cast<std::size_t>(src) * kMaxSamples + count);
        sweep[i].result = std::move(r);
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_inputs));
    CUDA_CHECK(cudaFree(d_samples));
    CUDA_CHECK(cudaFree(d_headers));

    return sweep;
}

}  // namespace sim

namespace viz {

struct ViewConfig {
    unsigned int width = 1280;
    unsigned int height = 720;
    float margin_left = 70.0f;
    float margin_right = 40.0f;
    float margin_top = 50.0f;
    float margin_bottom = 90.0f;
};

sf::Vector2f to_screen(const sim::Sample& sample, double max_x_ft, double max_y_ft, const ViewConfig& view) {
    const float usable_width = static_cast<float>(view.width) - view.margin_left - view.margin_right;
    const float usable_height = static_cast<float>(view.height) - view.margin_top - view.margin_bottom;
    const double x_norm = sample.x_ft / std::max(max_x_ft, 1.0);
    const double y_norm = sample.y_ft / std::max(max_y_ft, 1.0);
    const float x = view.margin_left + static_cast<float>(x_norm) * usable_width;
    const float ground_y = static_cast<float>(view.height) - view.margin_bottom;
    const float y = ground_y - static_cast<float>(y_norm) * usable_height;
    return {x, y};
}

sf::Color color_for_sweep_index(std::size_t index) {
    static const std::array<sf::Color, 8> palette = {
        sf::Color(255, 170, 80), sf::Color(120, 220, 120), sf::Color(255, 120, 160),
        sf::Color(180, 140, 255), sf::Color(120, 220, 255), sf::Color(255, 220, 120),
        sf::Color(160, 255, 210), sf::Color(255, 150, 110),
    };
    return palette[index % palette.size()];
}

void animate(const std::vector<sim::SweepEntry>& sweep, const sim::Inputs& reference_inputs) {
    if (sweep.empty()) return;

    ViewConfig view;
    sf::RenderWindow window(sf::VideoMode(view.width, view.height), "CUDA Golf Ball Flight Simulation");
    window.setFramerateLimit(60);

    sf::Font font;
    const bool has_font = font.loadFromFile("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf");

    double max_x_ft = 1.0;
    double max_y_ft = 1.0;
    double max_flight_time_s = 0.0;
    for (const sim::SweepEntry& entry : sweep) {
        max_flight_time_s = std::max(max_flight_time_s, entry.result.flight_time_s);
        for (const sim::Sample& sample : entry.result.trajectory) {
            max_x_ft = std::max(max_x_ft, sample.x_ft);
            max_y_ft = std::max(max_y_ft, sample.y_ft);
        }
    }
    max_x_ft *= 1.05;
    max_y_ft *= 1.15;

    std::vector<sf::VertexArray> full_paths;
    std::vector<sf::CircleShape> markers;
    full_paths.reserve(sweep.size());
    markers.reserve(sweep.size());

    std::size_t color_index = 0;
    for (const sim::SweepEntry& entry : sweep) {
        const sf::Color path_color = entry.is_reference ? sf::Color(0, 220, 255) : color_for_sweep_index(color_index++);
        sf::VertexArray path(sf::LineStrip, entry.result.trajectory.size());
        for (std::size_t i = 0; i < entry.result.trajectory.size(); ++i) {
            path[i].position = to_screen(entry.result.trajectory[i], max_x_ft, max_y_ft, view);
            path[i].color = path_color;
        }
        full_paths.push_back(path);

        sf::CircleShape marker(entry.is_reference ? 7.0f : 5.0f);
        marker.setOrigin(marker.getRadius(), marker.getRadius());
        marker.setFillColor(path_color);
        marker.setOutlineThickness(2.0f);
        marker.setOutlineColor(sf::Color::Black);
        markers.push_back(marker);
    }

    const float ground_y = static_cast<float>(view.height) - view.margin_bottom;
    sf::Vertex ground_line[] = {
        sf::Vertex(sf::Vector2f(view.margin_left, ground_y), sf::Color(100, 220, 100)),
        sf::Vertex(sf::Vector2f(static_cast<float>(view.width) - view.margin_right, ground_y), sf::Color(100, 220, 100)),
    };

    sf::Clock clock;
    while (window.isOpen()) {
        sf::Event event{};
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed) window.close();
        }

        const double elapsed_s = std::min<double>(clock.getElapsedTime().asSeconds(), max_flight_time_s);
        window.clear(sf::Color(18, 24, 38));
        window.draw(ground_line, 2, sf::Lines);

        for (std::size_t entry_index = 0; entry_index < sweep.size(); ++entry_index) {
            const auto& trajectory = sweep[entry_index].result.trajectory;
            if (trajectory.empty()) continue;
            std::size_t visible_count = 1;
            while (visible_count < trajectory.size() && trajectory[visible_count].time_s <= elapsed_s) ++visible_count;

            sf::VertexArray partial_path(sf::LineStrip, visible_count);
            for (std::size_t i = 0; i < visible_count; ++i) partial_path[i] = full_paths[entry_index][i];
            window.draw(partial_path);

            markers[entry_index].setPosition(to_screen(trajectory[visible_count - 1], max_x_ft, max_y_ft, view));
            window.draw(markers[entry_index]);
        }

        if (has_font) {
            sf::Text hud;
            hud.setFont(font);
            hud.setCharacterSize(22);
            hud.setFillColor(sf::Color::White);
            hud.setPosition(20.0f, 15.0f);
            hud.setString(
                "Ball speed: " + std::to_string(reference_inputs.ball_speed_mph) + " mph\n" +
                "Launch angle: " + std::to_string(reference_inputs.launch_angle_deg) + " deg\n" +
                "Reference spin: " + std::to_string(reference_inputs.spin_rate_rpm) + " rpm\n" +
                "CUDA spin sweep: +/-1000 rpm, step 250 rpm");
            window.draw(hud);
        }

        window.display();
        if (elapsed_s >= max_flight_time_s) clock.restart();
    }
}

}  // namespace viz

sim::Inputs read_inputs(int argc, char** argv) {
    sim::Inputs inputs;
    if (argc == 4) {
        inputs.ball_speed_mph = std::stod(argv[1]);
        inputs.launch_angle_deg = std::stod(argv[2]);
        inputs.spin_rate_rpm = std::stod(argv[3]);
        return inputs;
    }

    std::string line;
    std::cout << "Initial ball speed (mph) [" << inputs.ball_speed_mph << "]: ";
    std::getline(std::cin, line);
    if (!line.empty()) inputs.ball_speed_mph = std::stod(line);

    std::cout << "Launch angle (deg) [" << inputs.launch_angle_deg << "]: ";
    std::getline(std::cin, line);
    if (!line.empty()) inputs.launch_angle_deg = std::stod(line);

    std::cout << "Spin rate (rpm) [" << inputs.spin_rate_rpm << "]: ";
    std::getline(std::cin, line);
    if (!line.empty()) inputs.spin_rate_rpm = std::stod(line);
    return inputs;
}

void print_sweep_summary(const std::vector<sim::SweepEntry>& sweep) {
    std::cout << "\nCUDA spin sweep summary\n";
    std::cout << "spin_rpm,carry_yd,flight_time_s\n";
    for (const auto& entry : sweep) {
        std::cout << std::fixed << std::setprecision(2)
                  << entry.inputs.spin_rate_rpm << ","
                  << entry.result.carry_distance_yd << ","
                  << entry.result.flight_time_s;
        if (entry.is_reference) std::cout << " <- reference";
        std::cout << "\n";
    }
}

int main(int argc, char** argv) {
    try {
        const sim::Inputs inputs = read_inputs(argc, argv);
        auto base_sweep = sim::build_spin_sweep_inputs(inputs);

        float gpu_ms = 0.0f;
        int trajectories_computed = 0;
        auto gpu_sweep = sim::simulate_sweep_cuda(base_sweep, inputs, gpu_ms, trajectories_computed);

        std::cout << std::fixed << std::setprecision(3);

        print_sweep_summary(gpu_sweep);

        std::cout << "\nComputation timing\n";
        std::cout << "Kernel time = " << gpu_ms << " ms\n";
        std::cout << "Average time per trajectory = " << (gpu_ms / std::max(1, trajectories_computed)) << " ms\n";
        std::cout << "Trajectories computed = " << trajectories_computed << "\n";

        
        viz::animate(gpu_sweep, inputs);
        return EXIT_SUCCESS;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return EXIT_FAILURE;
    }
}
