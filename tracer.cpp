#include <SFML/Graphics.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace sim {

constexpr double kGravityFtPerSec2 = 32.174;
constexpr double kAirDensitySlugPerFt3 = 0.0023769;
constexpr double kBallRadiusFt = 1.68 / 24.0;
constexpr double kBallAreaFt2 = 3.14159265358979323846 * kBallRadiusFt * kBallRadiusFt;
constexpr double kBallMassSlug = 0.1012 / 32.174;
constexpr double kSecondsPerHour = 3600.0;
constexpr double kFeetPerMile = 5280.0;

using State = std::array<double, 4>;

struct Inputs {
    double ball_speed_mph = 128.0;
    double launch_angle_deg = 19.2;
    double spin_rate_rpm = 5400.0;
};

struct Sample {
    double time_s = 0.0;
    State state{};
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

double mph_to_ft_per_s(double mph) {
    return mph * kFeetPerMile / kSecondsPerHour;
}

double deg_to_rad(double deg) {
    return deg * 3.14159265358979323846 / 180.0;
}

double speed_ft_per_s(const State& state) {
    return std::sqrt(state[1] * state[1] + state[3] * state[3]);
}

double drag_coefficient(double speed_mph) {
    const double cd_high = 0.075;
    const double cd_low = 0.34;
    const double transition_mph = 130.0;
    const double width_mph = 15.0;
    return cd_high + (cd_low - cd_high) / (1.0 + std::exp((speed_mph - transition_mph) / width_mph));
}

double spin_decay_tau(double speed_mph) {
    return 5.0 + 7.0 / (1.0 + std::exp(-(speed_mph - 130.0) / 15.0));
}

double lift_coefficient(double spin_ratio) {
    const double slope = 1.8;
    const double max_lift = 0.57;
    const double value = slope * spin_ratio;
    return std::clamp(value, 0.0, max_lift);
}

State derivative(double time_s, const State& state, double spin_rate_rpm) {
    const double vx = state[1];
    const double vy = state[3];
    const double speed = speed_ft_per_s(state);
    const double speed_mph = speed * kSecondsPerHour / kFeetPerMile;
    const double cd = drag_coefficient(speed_mph);

    const double omega0 = spin_rate_rpm * 2.0 * 3.14159265358979323846 / 60.0;
    const double tau = spin_decay_tau(speed_mph);
    const double omega = omega0 * std::exp(-time_s / tau);
    const double spin_ratio = omega * kBallRadiusFt / std::max(speed, 1e-6);
    const double cl = lift_coefficient(spin_ratio);

    const double drag_force = 0.5 * kAirDensitySlugPerFt3 * kBallAreaFt2 * cd * speed * speed;
    const double lift_force = 0.5 * kAirDensitySlugPerFt3 * kBallAreaFt2 * cl * speed * speed;

    const double drag_x = -drag_force * vx / std::max(speed, 1e-6);
    const double drag_y = -drag_force * vy / std::max(speed, 1e-6);
    const double lift_x = -lift_force * vy / std::max(speed, 1e-6);
    const double lift_y = lift_force * vx / std::max(speed, 1e-6);

    return State{
        vx,
        (drag_x + lift_x) / kBallMassSlug,
        vy,
        -kGravityFtPerSec2 + (drag_y + lift_y) / kBallMassSlug,
    };
}

State add_scaled(const State& a, const State& b, double scale) {
    State out{};
    for (std::size_t i = 0; i < out.size(); ++i) {
        out[i] = a[i] + scale * b[i];
    }
    return out;
}

double max_abs_diff(const State& a, const State& b) {
    double max_value = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        max_value = std::max(max_value, std::abs(a[i] - b[i]));
    }
    return max_value;
}

struct StepResult {
    State next_state{};
    State embedded_state{};
};

StepResult rk45_step(double time_s, const State& state, double step_s, double spin_rate_rpm) {
    const State k1 = derivative(time_s, state, spin_rate_rpm);
    const State k2 = derivative(time_s + step_s * (1.0 / 5.0), add_scaled(state, k1, step_s * (1.0 / 5.0)), spin_rate_rpm);

    State stage = state;
    for (std::size_t i = 0; i < stage.size(); ++i) {
        stage[i] = state[i] + step_s * ((3.0 / 40.0) * k1[i] + (9.0 / 40.0) * k2[i]);
    }
    const State k3 = derivative(time_s + step_s * (3.0 / 10.0), stage, spin_rate_rpm);

    for (std::size_t i = 0; i < stage.size(); ++i) {
        stage[i] = state[i] + step_s * ((44.0 / 45.0) * k1[i] + (-56.0 / 15.0) * k2[i] + (32.0 / 9.0) * k3[i]);
    }
    const State k4 = derivative(time_s + step_s * (4.0 / 5.0), stage, spin_rate_rpm);

    for (std::size_t i = 0; i < stage.size(); ++i) {
        stage[i] = state[i] + step_s * ((19372.0 / 6561.0) * k1[i] + (-25360.0 / 2187.0) * k2[i] +
                                        (64448.0 / 6561.0) * k3[i] + (-212.0 / 729.0) * k4[i]);
    }
    const State k5 = derivative(time_s + step_s * (8.0 / 9.0), stage, spin_rate_rpm);

    for (std::size_t i = 0; i < stage.size(); ++i) {
        stage[i] = state[i] + step_s * ((9017.0 / 3168.0) * k1[i] + (-355.0 / 33.0) * k2[i] +
                                        (46732.0 / 5247.0) * k3[i] + (49.0 / 176.0) * k4[i] +
                                        (-5103.0 / 18656.0) * k5[i]);
    }
    const State k6 = derivative(time_s + step_s, stage, spin_rate_rpm);

    State fifth_order{};
    for (std::size_t i = 0; i < fifth_order.size(); ++i) {
        fifth_order[i] = state[i] + step_s * ((35.0 / 384.0) * k1[i] + (500.0 / 1113.0) * k3[i] +
                                              (125.0 / 192.0) * k4[i] + (-2187.0 / 6784.0) * k5[i] +
                                              (11.0 / 84.0) * k6[i]);
    }

    const State k7 = derivative(time_s + step_s, fifth_order, spin_rate_rpm);

    State fourth_order{};
    for (std::size_t i = 0; i < fourth_order.size(); ++i) {
        fourth_order[i] = state[i] + step_s * ((5179.0 / 57600.0) * k1[i] + (7571.0 / 16695.0) * k3[i] +
                                               (393.0 / 640.0) * k4[i] + (-92097.0 / 339200.0) * k5[i] +
                                               (187.0 / 2100.0) * k6[i] + (1.0 / 40.0) * k7[i]);
    }

    return StepResult{fifth_order, fourth_order};
}

Sample interpolate_ground_hit(const Sample& above, const Sample& below) {
    const double y0 = above.state[2];
    const double y1 = below.state[2];
    double fraction = 0.0;
    if (std::abs(y1 - y0) > 1e-12) {
        fraction = (0.0 - y0) / (y1 - y0);
    }
    fraction = std::clamp(fraction, 0.0, 1.0);

    Sample hit{};
    hit.time_s = above.time_s + fraction * (below.time_s - above.time_s);
    for (std::size_t i = 0; i < hit.state.size(); ++i) {
        hit.state[i] = above.state[i] + fraction * (below.state[i] - above.state[i]);
    }
    hit.state[2] = 0.0;
    return hit;
}

Result simulate(const Inputs& inputs) {
    const double speed0 = mph_to_ft_per_s(inputs.ball_speed_mph);
    const double theta = deg_to_rad(inputs.launch_angle_deg);

    Sample sample{};
    sample.time_s = 0.0;
    sample.state = State{
        0.0,
        speed0 * std::cos(theta),
        0.01,
        speed0 * std::sin(theta),
    };

    Result result;
    result.trajectory.push_back(sample);

    double time_s = 0.0;
    double step_s = 0.01;
    constexpr double end_time_s = 20.0;
    constexpr double abs_tol = 1e-10;
    constexpr double rel_tol = 1e-8;
    constexpr double min_step_s = 1e-6;
    constexpr double max_step_s = 0.05;

    while (time_s < end_time_s) {
        if (step_s < min_step_s) {
            throw std::runtime_error("Adaptive solver step size underflowed.");
        }
        if (time_s + step_s > end_time_s) {
            step_s = end_time_s - time_s;
        }

        const StepResult step = rk45_step(time_s, sample.state, step_s, inputs.spin_rate_rpm);

        double scale = 0.0;
        for (std::size_t i = 0; i < sample.state.size(); ++i) {
            scale = std::max(scale, std::abs(step.next_state[i]));
        }
        const double tolerance = abs_tol + rel_tol * std::max(1.0, scale);
        const double error = max_abs_diff(step.next_state, step.embedded_state);

        if (error <= tolerance) {
            Sample next_sample{};
            next_sample.time_s = time_s + step_s;
            next_sample.state = step.next_state;

            if (next_sample.state[2] <= 0.0) {
                const Sample hit = interpolate_ground_hit(sample, next_sample);
                result.trajectory.push_back(hit);
                result.carry_distance_ft = hit.state[0];
                result.carry_distance_yd = hit.state[0] / 3.0;
                result.flight_time_s = hit.time_s;
                return result;
            }

            result.trajectory.push_back(next_sample);
            sample = next_sample;
            time_s = next_sample.time_s;

            double factor = 1.5;
            if (error > 0.0) {
                factor = 0.9 * std::pow(tolerance / error, 0.2);
            }
            step_s = std::clamp(step_s * factor, min_step_s, max_step_s);
        } else {
            const double factor = 0.9 * std::pow(tolerance / std::max(error, 1e-16), 0.25);
            step_s = std::clamp(step_s * std::max(0.1, factor), min_step_s, max_step_s);
        }
    }

    const Sample& last = result.trajectory.back();
    result.carry_distance_ft = last.state[0];
    result.carry_distance_yd = last.state[0] / 3.0;
    result.flight_time_s = last.time_s;
    return result;
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

    const double x_norm = sample.state[0] / std::max(max_x_ft, 1.0);
    const double y_norm = sample.state[2] / std::max(max_y_ft, 1.0);

    const float x = view.margin_left + static_cast<float>(x_norm) * usable_width;
    const float ground_y = static_cast<float>(view.height) - view.margin_bottom;
    const float y = ground_y - static_cast<float>(y_norm) * usable_height;
    return {x, y};
}

sf::Color color_for_sweep_index(std::size_t index) {
    static const std::array<sf::Color, 8> palette = {
        sf::Color(255, 170, 80),
        sf::Color(120, 220, 120),
        sf::Color(255, 120, 160),
        sf::Color(180, 140, 255),
        sf::Color(120, 220, 255),
        sf::Color(255, 220, 120),
        sf::Color(160, 255, 210),
        sf::Color(255, 150, 110),
    };
    return palette[index % palette.size()];
}

void animate(const std::vector<sim::SweepEntry>& sweep, const sim::Inputs& reference_inputs) {
    if (sweep.empty()) {
        return;
    }

    ViewConfig view;
    sf::RenderWindow window(sf::VideoMode(view.width, view.height), "Golf Ball Flight Baseline");
    window.setFramerateLimit(60);

    sf::Font font;
    const bool has_font = font.loadFromFile("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf");

    double max_x_ft = 1.0;
    double max_y_ft = 1.0;
    double max_flight_time_s = 0.0;
    for (const sim::SweepEntry& entry : sweep) {
        max_flight_time_s = std::max(max_flight_time_s, entry.result.flight_time_s);
        for (const sim::Sample& sample : entry.result.trajectory) {
            max_x_ft = std::max(max_x_ft, sample.state[0]);
            max_y_ft = std::max(max_y_ft, sample.state[2]);
        }
    }
    max_x_ft *= 1.05;
    max_y_ft *= 1.15;

    std::vector<sf::VertexArray> full_paths;
    full_paths.reserve(sweep.size());
    std::vector<sf::CircleShape> markers;
    markers.reserve(sweep.size());

    std::size_t color_index = 0;
    for (const sim::SweepEntry& entry : sweep) {
        const sf::Color path_color = entry.is_reference ? sf::Color(0, 200, 255) : color_for_sweep_index(color_index++);
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
            if (event.type == sf::Event::Closed) {
                window.close();
            }
        }

        const double elapsed_s = std::min<double>(clock.getElapsedTime().asSeconds(), max_flight_time_s);

        window.clear(sf::Color(18, 24, 38));
        window.draw(ground_line, 2, sf::Lines);

        for (std::size_t entry_index = 0; entry_index < sweep.size(); ++entry_index) {
            const auto& trajectory = sweep[entry_index].result.trajectory;
            std::size_t visible_count = 1;
            while (visible_count < trajectory.size() && trajectory[visible_count].time_s <= elapsed_s) {
                ++visible_count;
            }

            sf::VertexArray partial_path(sf::LineStrip, visible_count);
            for (std::size_t i = 0; i < visible_count; ++i) {
                partial_path[i] = full_paths[entry_index][i];
            }
            window.draw(partial_path);

            const sim::Sample& current_sample = trajectory[visible_count - 1];
            markers[entry_index].setPosition(to_screen(current_sample, max_x_ft, max_y_ft, view));
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
                "Sweep: +/-1000 rpm in 250 rpm steps");
            window.draw(hud);
        }

        window.display();

        if (elapsed_s >= max_flight_time_s) {
            clock.restart();
        }
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

    std::cout << "Initial ball speed (mph) [" << inputs.ball_speed_mph << "]: ";
    std::string line;
    std::getline(std::cin, line);
    if (!line.empty()) {
        inputs.ball_speed_mph = std::stod(line);
    }

    std::cout << "Launch angle (deg) [" << inputs.launch_angle_deg << "]: ";
    std::getline(std::cin, line);
    if (!line.empty()) {
        inputs.launch_angle_deg = std::stod(line);
    }

    std::cout << "Spin rate (rpm) [" << inputs.spin_rate_rpm << "]: ";
    std::getline(std::cin, line);
    if (!line.empty()) {
        inputs.spin_rate_rpm = std::stod(line);
    }

    return inputs;
}

void print_results(const sim::Result& result) {
    std::cout << std::fixed << std::setprecision(6);
    // std::cout << "time_s,x_ft,y_ft,vx_ft_per_s,vy_ft_per_s\n";
    // for (const sim::Sample& sample : result.trajectory) {
    //     std::cout << sample.time_s << ","
    //               << sample.state[0] << ","
    //               << sample.state[2] << ","
    //               << sample.state[1] << ","
    //               << sample.state[3] << "\n";
    // }

    std::cout << std::setprecision(2);
    std::cout << "Carry distance = " << result.carry_distance_ft
              << " ft = " << result.carry_distance_yd << " yd\n";
    std::cout << "Flight time = " << result.flight_time_s << " s\n";
}

std::vector<sim::SweepEntry> build_spin_sweep(const sim::Inputs& reference_inputs) {
    std::vector<sim::SweepEntry> sweep;
    for (int offset = -1000; offset <= 1000; offset += 250) {
        sim::Inputs inputs = reference_inputs;
        inputs.spin_rate_rpm = std::max(0.0, reference_inputs.spin_rate_rpm + static_cast<double>(offset));
        sim::SweepEntry entry;
        entry.inputs = inputs;
        entry.result = sim::simulate(inputs);
        entry.is_reference = std::abs(inputs.spin_rate_rpm - reference_inputs.spin_rate_rpm) < 1e-9;
        sweep.push_back(entry);
    }
    return sweep;
}

void print_sweep_summary(const std::vector<sim::SweepEntry>& sweep) {
    std::cout << "\nSpin sweep summary\n";
    std::cout << "spin_rpm,carry_yd,flight_time_s\n";
    for (const sim::SweepEntry& entry : sweep) {
        std::cout << std::fixed << std::setprecision(2)
                  << entry.inputs.spin_rate_rpm << ","
                  << entry.result.carry_distance_yd << ","
                  << entry.result.flight_time_s;
        if (entry.is_reference) {
            std::cout << " <- reference";
        }
        std::cout << "\n";
    }
}

void print_timing_summary(
    const std::chrono::steady_clock::time_point& simulation_start,
    const std::chrono::steady_clock::time_point& simulation_end,
    std::size_t trajectories_computed) {
    using milliseconds = std::chrono::duration<double, std::milli>;

    const milliseconds total_ms = simulation_end - simulation_start;
    const double average_ms = trajectories_computed > 0
        ? total_ms.count() / static_cast<double>(trajectories_computed)
        : 0.0;

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "\nComputation timing\n";
    std::cout << "Total simulation time = " << total_ms.count() << " ms\n";
    std::cout << "Average time per trajectory = " << average_ms << " ms\n";
    std::cout << "Trajectories computed = " << trajectories_computed << "\n";
}

int main(int argc, char** argv) {
    try {
        const sim::Inputs inputs = read_inputs(argc, argv);
        const auto simulation_start = std::chrono::steady_clock::now();
        const sim::Result result = sim::simulate(inputs);
        const std::vector<sim::SweepEntry> sweep = build_spin_sweep(inputs);
        const auto simulation_end = std::chrono::steady_clock::now();
        print_results(result);
        print_sweep_summary(sweep);
        print_timing_summary(simulation_start, simulation_end, sweep.size() + 1);
        viz::animate(sweep, inputs);
        return EXIT_SUCCESS;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return EXIT_FAILURE;
    }
}
