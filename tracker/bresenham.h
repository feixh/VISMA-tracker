//
// Created by visionlab on 2/8/18.
// Reference:
// https://github.com/ssloy/tinyrenderer/wiki/Lesson-1:-Bresenham%E2%80%99s-Line-Drawing-Algorithm

#pragma once

// stl
#include <array>

namespace feh {

/// \brief: Bresenham Line Iterator for 1-dim search.
class BresenhamLineIterator {
public:
    BresenhamLineIterator(int x1, int y1, int x2, int y2);
    BresenhamLineIterator& operator++();
    BresenhamLineIterator operator++(int);
    int size() const { return count_; }
    int x() const { return steep_ ? y_ : x_; }
    int y() const { return steep_ ? x_ : y_; }
private:
    int dx_, dy_, dx2_, dy2_; // difference of target & starting points
    int x_, y_;   // current x, y
    int s_;    // sign of y
    int err_;
    bool steep_;
    int count_;
};

}
