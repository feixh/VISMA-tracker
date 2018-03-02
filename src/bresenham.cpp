//
// Created by visionlab on 2/8/18.
//
#include "bresenham.h"

// 3rd party
#include "opencv2/imgproc.hpp"
#include "glog/logging.h"

namespace feh {

BresenhamLineIterator::BresenhamLineIterator(int x1, int y1, int x2, int y2) {
    steep_ = false;
    if (abs(x1-x2) < abs(y1-y2)) {
        std::swap(x1, y1);
        std::swap(x2, y2);
        steep_ = true;
    }
    if (x1 > x2) {
        std::swap(x1, x2);
        std::swap(y1, y2);
    }
    dx_ = x2-x1;    // positive by construction
    count_ = dx_+1;
    s_ = y2 > y1 ? 1 : -1;
    dy_ = s_ * (y2 - y1);    // ensure dy is positive
    dx2_ = (dx_ << 1);
    dy2_ =  (dy_ << 1);
    err_ = 0;
    x_ = x1;
    y_ = y1;
}

BresenhamLineIterator& BresenhamLineIterator::operator++() {
    if (err_ > dx_) {
        y_ += s_;
        err_ -= dx2_;
    }
    ++x_;
    err_ += dy2_;
    return *this;
}

BresenhamLineIterator BresenhamLineIterator::operator++(int) {
    BresenhamLineIterator it = *this;
    ++(*this);
    return it;
}

}

