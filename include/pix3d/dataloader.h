#include "common/eigen_alias.h"

#include <vector>

#include "opencv2/core.hpp"
#include "igl/readOBJ.h"
#include "folly/json.h"
#include "folly/FileUtil.h"

namespace feh {

struct Pix3dPacket {
    cv::Mat _img, _mask;
    Mat3 _rot_mat;
    Vec3 _trans_mat, _cam_position; // make the names consistent with pix3d json file
    ftype _focal_length, _inplane_rotation;
    Vec4i _bbox;
};

class Pix3dLoader {
public:
    Pix3dLoader(const std::string &dataroot): _dataroot(dataroot) {
        // load json file

        std::string json_path{dataroot + "/pix3d.json"};
        std::string content;
        folly::readFile(json_path.c_str(), content);
        _json = folly::parseJson(folly::json::stripComments(content));
    }

    Pix3dPacket GrabPacket(int i) {
        const folly::dynamic record = _json[i];
        std::cout << record;
        // Eigen::MatrixXf V;
        // Eigen::MatrixXi F;
        // bool success = igl::readOBJ(obj_file, V, F);
    }




private:
    std::string _dataroot;
    folly::dynamic _json;
    

};

} // namespace feh
