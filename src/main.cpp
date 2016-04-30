#include <algorithm>
#include <boost/program_options.hpp>
#include <iostream>
#include <memory>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace po = boost::program_options;

using namespace std;

using cv::Canny;
using cv::GaussianBlur;
using cv::Mat;
using cv::Point;
using cv::Point2f;
using cv::Scalar;
using cv::Size;
using cv::addWeighted;
using cv::circle;
using cv::contourArea;
using cv::cvtColor;
using cv::destroyAllWindows;
using cv::findContours;
using cv::getPerspectiveTransform;
using cv::getStructuringElement;
using cv::imread;
using cv::imshow;
using cv::morphologyEx;
using cv::namedWindow;
using cv::resize;
using cv::setMouseCallback;
using cv::updateWindow;
using cv::waitKey;
using cv::warpPerspective;

struct AnchoredRect {
  Point tl;
  Point tr;
  Point br;
  Point bl;

  AnchoredRect() {
  }

  AnchoredRect(const vector<Point>& points) {
    if (points.size() != 4) {
      throw invalid_argument("AnchoredRect constructor did not receive 4 "
                             "points.");
    }
    auto sum = [] (const Point& a, const Point& b) {
          return a.x + a.y < b.x + b.y;
        };
    auto diff = [] (const Point& a, const Point& b) {
          return a.x - a.y < b.x - b.y;
        };
    tl = *min_element(points.begin(), points.end(), sum);
    br = *max_element(points.begin(), points.end(), sum);
    tr = *max_element(points.begin(), points.end(), diff);
    bl = *min_element(points.begin(), points.end(), diff);
  }

  operator vector<Point>() const {
    return {tl, tr, br, bl};
  }

  operator vector<Point2f>() const {
    vector<Point2f> src;
    for (const Point& p : {tl, tr, br, bl}) {
      src.push_back(Point2f{static_cast<float>(p.x),
                            static_cast<float>(p.y)});
    }
    return src;
  }

  operator Mat() const {
    Mat contour;
    contour.push_back(tl);
    contour.push_back(tr);
    contour.push_back(br);
    contour.push_back(bl);
    return contour;
  }
};

struct ExtractionState {
  Mat original;
  AnchoredRect contour;
};

struct ProgramState {
  ExtractionState state;
  Mat guiImage;
  int selectedPoint;
};

const string WINDOW_NAME = "DocScanner";
const int WINDOW_HEIGHT = 800;
const double ANCHOR_RADIUS = 10.0;
const Scalar ANCHOR_COLOR = Scalar(0, 0, 255);
const double ANCHOR_ALPHA = 0.5;

bool parseArguments(int argc, char* argv[],
                    shared_ptr<string> inputFilename,
                    shared_ptr<string> outputFilename) {
  if (!inputFilename) {
    throw invalid_argument("inputFilename is a nullptr.");
  }
  if (!outputFilename) {
    throw invalid_argument("outputFilename is a nullptr.");
  }

  po::options_description desc("Allowed options");
  desc.add_options()
    ("input", po::value<string>(inputFilename.get()),
     "Filename of the image to process")
    ("output", po::value<string>(outputFilename.get()),
     "Output filename")
  ;

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  if (!vm.count("input")) {
    cerr << "input argument has not been provided." << endl;
    return false;
  } else if (!vm.count("output")) {
    cerr << "output argument has not been provided." << endl;
    return false;
  } else {
    return true;
  }
}

Mat resizeImage(const Mat image, const double ratio) {
  Mat resized;
  resize(image, resized, Size(image.size[1] / ratio,
                              image.size[0] / ratio));
  return resized;
}

Mat resizeImageWidth(const Mat image, const int width) {
  const double ratio = image.size[1] / double(width);
  Mat resized;
  resize(image, resized, Size(width,
                              image.size[0] / ratio));
  return resized;
}

Mat resizeImageHeight(const Mat image, const int height) {
  const double ratio = image.size[0] / double(height);
  Mat resized;
  resize(image, resized, Size(image.size[1] / ratio,
                              height));
  return resized;
}

Mat resizeContour(const Mat contour, const double ratio) {
  Mat resized;
  for (auto iter = contour.begin<Point2f>();
       iter != contour.end<Point2f>();
       ++iter) {
    resized.push_back(Point2f(ratio * (*iter).x, ratio * (*iter).y));
  }
  return resized;
}

Mat edgeImage(const Mat image) {
  Mat gray;
  cvtColor(image, gray, cv::COLOR_BGR2GRAY);
  GaussianBlur(gray, gray, Size(5, 5), 0);
  Canny(gray, gray, 75, 200);
  return gray;
}

Mat closeMorphologically(const Mat image, const int kernel_size) {
  Mat kernel = getStructuringElement(
      cv::MORPH_RECT,
      Size(kernel_size, kernel_size));

  Mat dst;
  morphologyEx(image, dst, cv::MORPH_CLOSE, kernel);
  return dst;
}

vector<Mat> findContours(const Mat edgeImage) {
  Mat copy = edgeImage;
  vector<Mat> contours;
  Mat hierarchy;
  findContours(copy, contours, hierarchy, cv::RETR_LIST,
               cv::CHAIN_APPROX_SIMPLE);

  sort(contours.begin(), contours.end(),
       [] (const Mat& a, const Mat& b) {
         return contourArea(a) > contourArea(b);
       });
  return contours;
}

void approxContour(const Mat& input, Mat& output) {
  double length = arcLength(input, true);
  approxPolyDP(input, output, 0.02 * length, true);
}

bool isContourLikeRectangle(const Mat& contour) {
  Mat output;
  approxContour(contour, output);
  return output.size[0] == 4;
}

double distance(const Point& a, const Point& b) {
  return sqrt(pow(a.x - b.x, 2.0) + pow(a.y - b.y, 2.0));
}

Mat fourPointTransform(const Mat image, const AnchoredRect& rect) {
  float maxWidth = max(distance(rect.br, rect.bl), distance(rect.tr, rect.tl));
  float maxHeight = max(distance(rect.tr, rect.br), distance(rect.tl, rect.bl));

  Mat transform = getPerspectiveTransform(
      vector<Point2f>(rect),
      vector<Point2f>{{0, 0}, {maxWidth - 1, 0}, {maxWidth - 1, maxHeight - 1},
                      {0, maxHeight - 1}});
  Mat warped;
  warpPerspective(image, warped, transform, {static_cast<int>(maxWidth),
                                             static_cast<int>(maxHeight)});

  return warped;
}

vector<Point> matToPoints(const Mat& input) {
  return vector<Point>(input.begin<Point>(), input.end<Point>());
}

Mat applyBWThreshold(const Mat& image) {
  Mat output;
  cvtColor(image, output, cv::COLOR_BGR2GRAY);
  adaptiveThreshold(output, output, 251, cv::ADAPTIVE_THRESH_MEAN_C,
                    cv::THRESH_BINARY, 25, 10);
  return output;
}

// Finds a best match for a contour of a document.
// Returns true and writes the contour to rect if the contour could be found.
// Otherwise rect is the contour of the entire input image.
bool extractDocument(const Mat& imageWithDocument,
                     AnchoredRect* rect) {
  const double too_small_image_threshold = 0.1;
  const double processed_image_width = 500.0;
  const int morphology_kernel_size = 5;
  const AnchoredRect fullImageRect = AnchoredRect(
      {{0, 0}, {imageWithDocument.size[1], 0},
      {imageWithDocument.size[1], imageWithDocument.size[0]},
      {0, imageWithDocument.size[0]}});

  Mat image = resizeImage(
      imageWithDocument,
      imageWithDocument.size[1] / processed_image_width);
  Mat edged = edgeImage(image);
  Mat closed = closeMorphologically(edged, morphology_kernel_size);
  vector<Mat> contours = findContours(closed);
  vector<Mat> bestContours;
  remove_copy_if(contours.begin(), contours.end(),
                 back_inserter(bestContours),
                 [] (const Mat& c) { return !isContourLikeRectangle(c); });

  if (bestContours.empty()) {
    cout << "Couldn't find any contour resembling a rectangle." << endl;
    *rect = fullImageRect;
    return false;
  }

  Mat pageContour = bestContours[0];
  approxContour(pageContour, pageContour);
  pageContour = resizeContour(
      pageContour,
      imageWithDocument.size[1] / image.size[1]);
  *rect = AnchoredRect(matToPoints(pageContour));

  double contourToInputWidthRatio =
      distance(rect->bl, rect->br) / imageWithDocument.size[1];
  double contourToInputHeightRatio =
      distance(rect->tl, rect->bl) / imageWithDocument.size[0];
  if (contourToInputWidthRatio < too_small_image_threshold ||
      contourToInputHeightRatio < too_small_image_threshold) {
    cout << "Couldn't find a big enough contour." << endl;
    *rect = fullImageRect;
    return false;
  }
  return true;
}

void drawFilledCircle(const Mat& img, Point center, double radius,
                      Mat* output) {
  circle(*output, center, static_cast<int>(radius), ANCHOR_COLOR, -1, 0);
}

void drawLine(const Mat& img, Point a, Point b, const double lineThickness,
              Mat* output) {
  line(*output, a, b, ANCHOR_COLOR, lineThickness, 3);
}

Mat drawAnchoredRect(const Mat& img, const AnchoredRect& rect,
                     const double lineThickness,
                     const double anchorRadius) {
  Mat output;
  img.copyTo(output);

  drawLine(img, rect.tl, rect.tr, lineThickness, &output);
  drawLine(img, rect.tr, rect.br, lineThickness, &output);
  drawLine(img, rect.br, rect.bl, lineThickness, &output);
  drawLine(img, rect.bl, rect.tl, lineThickness, &output);
  for (const Point& p : vector<Point>(rect)) {
    drawFilledCircle(img, p, anchorRadius, &output);
  }
  addWeighted(img, 1.0 - ANCHOR_ALPHA, output, ANCHOR_ALPHA, 0.0, output);
  return output;
}

Mat drawGUIBackground(const ExtractionState& state) {
  Mat extraction = fourPointTransform(state.original, state.contour);
  Mat scaledExtraction = resizeImageHeight(extraction, state.original.size[0]);

  Mat concatenated;
  hconcat(state.original, scaledExtraction, concatenated);
  return resizeImageHeight(concatenated, WINDOW_HEIGHT);
}

void drawContourAndUpdateGUI(const ProgramState& state) {
  const double lineThickness = 3.0;
  const double ratio = state.state.original.size[0] / double(WINDOW_HEIGHT);

  auto scaledContour = vector<Point>(state.state.contour);
  for (Point& p : scaledContour) {
    p *= (1.0 / ratio);
  }

  Mat withAnchors = drawAnchoredRect(
      state.guiImage,
      scaledContour,
      lineThickness,
      ANCHOR_RADIUS);
  imshow(WINDOW_NAME, withAnchors);
  updateWindow(WINDOW_NAME);
}

Point sanitizeMousePointInput(const Point& input,
                              const ExtractionState& state) {
  return {min(max(input.x, 0), state.original.size[1]),
          min(max(input.y, 0), state.original.size[0])};
}

void onMouse(int event, int x, int y, int flags, void* data) {
  ProgramState& programState = *((ProgramState*) data);
  ExtractionState& state = programState.state;
  const double guiScaledown = state.original.size[0] / double(WINDOW_HEIGHT);
  if (event == cv::EVENT_LBUTTONDOWN) {
    vector<Point> contour(state.contour);
    for (size_t i = 0; i < contour.size(); ++i) {
      if (distance(contour[i] * (1.0 / guiScaledown), Point{x, y}) <
          ANCHOR_RADIUS) {
        programState.selectedPoint = i;
        break;
      }
    }
  }

  if ((event == cv::EVENT_MOUSEMOVE || event == cv::EVENT_LBUTTONDOWN) &&
      (flags & cv::EVENT_FLAG_LBUTTON) &&
      programState.selectedPoint != -1) {
    vector<Point> contour(state.contour);
    const double guiScaledown = state.original.size[0] / double(WINDOW_HEIGHT);
    contour[programState.selectedPoint] =
      sanitizeMousePointInput(Point{x, y} * guiScaledown, state);
    state.contour = AnchoredRect(contour);
    drawContourAndUpdateGUI(programState);
  } else if (event == cv::EVENT_LBUTTONUP && programState.selectedPoint != -1) {
    programState.guiImage = drawGUIBackground(programState.state);
    programState.selectedPoint = -1;
    drawContourAndUpdateGUI(programState);
  }
}

void initGUI(ProgramState* state) {
  state->guiImage = drawGUIBackground(state->state);
  drawContourAndUpdateGUI(*state);
  setMouseCallback(WINDOW_NAME, onMouse, state);
}

int main(int argc, char* argv[]) {
  try {
    shared_ptr<string> inputFilename = make_shared<string>();
    shared_ptr<string> outputFilename = make_shared<string>();
    if (!parseArguments(argc, argv, inputFilename, outputFilename)) {
      return 1;
    }

    Mat original = imread(*inputFilename);
    AnchoredRect rect;
    extractDocument(original, &rect);

    ProgramState programState{{original, rect}, {}, -1};
    initGUI(&programState);

    bool shouldEnd = false;
    while (!shouldEnd) {
      switch (waitKey(0)) {
        case '1':
          programState.selectedPoint = 0;
          break;
        case '2':
          programState.selectedPoint = 1;
          break;
        case '3':
          programState.selectedPoint = 2;
          break;
        case '4':
          programState.selectedPoint = 3;
          break;
        case 'q':
          shouldEnd = true;
          break;
        case '\n':
          const Mat extraction = fourPointTransform(programState.state.original,
                                                    programState.state.contour);
          if(!imwrite(*outputFilename, extraction)) {
            cerr << "Couldn't save output image to: " << *outputFilename
                 << endl;
          }
          shouldEnd = true;
          break;
      }
    }
    destroyAllWindows();

    return 0;
  } catch(exception& e) {
    cerr << "Exception thrown: " << e.what() << "\n";
    return 1;
  } catch(...) {
    cerr << "Exception of unknown type has been thrown!\n";
    return 1;
  }
}
