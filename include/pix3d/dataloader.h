// Dataset loader for Pix3d dataset.
// stl
#include <vector>
#include <array>

// 3rdparty
#include "opencv2/highgui.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "folly/json.h"
#include "folly/FileUtil.h"
#include "igl/readOBJ.h"
#include "sophus/se3.hpp"

// feh
#include "eigen_alias.h"
#include "utils.h"

namespace feh {

struct Pix3dPacket {
    Pix3dPacket(const std::string &dataroot, const folly::dynamic &record) {
        // load image & mask
        _img = cv::imread(dataroot + record["img"].asString());
        _mask = cv::imread(dataroot + record["mask"].asString());
        _bbox = GetVectorFromDynamic<int, 4>(record, "bbox");

        // load object pose
        _go = Sophus::SE3<ftype>{
            GetMatrixFromDynamic<ftype, 3, 3>(record, "rot_mat", JsonMatLayout::RowMajor),
            GetVectorFromDynamic<ftype, 3>(record, "trans_mat")};

        // camera intrinsics
        _shape = GetVectorFromDynamic<int, 2>(record, "img_size");  // input layout: [width, height]
        std::swap(_shape[0], _shape[1]);    // shape layout: [height, width]
        _focal_length = record["focal_length"].asDouble() / 32.0 * _shape[1];     // convert from mm to pixels


        // construct camera pose from position and in-plane rotation
        _cam_position = GetVectorFromDynamic<ftype, 3>(record, "cam_position");
        _inplane_rotation  = record["inplane_rotation"].asDouble();
        Vec3 dir = Vec3::Zero() - _cam_position;
        dir /= dir.norm();
        _gc = Sophus::SE3<ftype>(Sophus::SO3<ftype>::exp(-_inplane_rotation * dir), _cam_position);

        // load CAD model
        bool success = igl::readOBJ(dataroot + record["model"].asString(), _V, _F);
        assert(success);
    }
    cv::Mat _img, _mask;
    Sophus::SE3<ftype> _go;  // object pose, applied to object directly
    Sophus::SE3<ftype> _gc;  // camera pose FIXME: add more details later
    Vec3 _cam_position; // make the names consistent with pix3d json file
    ftype _inplane_rotation;    // assuming object sitting at the origin, camera is looking at the object center with an inplane rotation
    ftype _focal_length;
    Vec4i _bbox;
    MatX _V;    // vertices &
    MatXi _F;   // faces of CAD models
    Eigen::Matrix<int, 2, 1> _shape;  // shape of images: [rows, cols]
};

class Pix3dLoader {
public:
    Pix3dLoader(const std::string &dataroot): _dataroot(dataroot+"/") {
        // load json file

        std::string json_path{dataroot + "/pix3d.json"};
        std::string content;
        folly::readFile(json_path.c_str(), content);
        std::cout << TermColor::cyan << "parsing json file ..." << TermColor::endl;
        _json = folly::parseJson(folly::json::stripComments(content));
    }

    /// \brief: Grab datum by indexing the json file.
    /// \param i: Index of the datum.
    Pix3dPacket GrabPacket(int i) {
        return {_dataroot, _json[i]};
    }

    /// \brief: Grab datum by filename of the datum.
    Pix3dPacket GrabPacket(const std::string &path) {
        for (int i = 0; i < _json.size(); ++i) {
            if (_json[i]["img"] == path) {
                return {_dataroot, _json[i]};
            }
        }
        throw std::out_of_range("failed to find datum at " + path);
    }


private:
    std::string _dataroot;
    folly::dynamic _json;

};

} // namespace feh
