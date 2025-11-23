#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <vector>

namespace mycnn {

struct Shape {
    int n{0};
    int h{0};
    int w{0};
    int c{0};
};

class Tensor {
public:
    Tensor() = default;
    Tensor(Shape shape) : shape_(shape), data_(shape.n * shape.h * shape.w * shape.c, 0.0f) {}

    float &operator()(int n, int h, int w, int c) {
        return data_[offset(n, h, w, c)];
    }

    const float &operator()(int n, int h, int w, int c) const {
        return data_[offset(n, h, w, c)];
    }

    std::vector<float> &data() { return data_; }
    const std::vector<float> &data() const { return data_; }

    Shape shape() const { return shape_; }

    size_t size() const { return data_.size(); }

    void fill(float value) { std::fill(data_.begin(), data_.end(), value); }

private:
    size_t offset(int n, int h, int w, int c) const {
        assert(n < shape_.n && h < shape_.h && w < shape_.w && c < shape_.c);
        return ((static_cast<size_t>(n) * shape_.h + h) * shape_.w + w) * shape_.c + c;
    }

    Shape shape_{};
    std::vector<float> data_{};
};

} // namespace mycnn
