using OpenCVForUnity;
using System.Collections.Generic;
using UnityEngine;
using Rect = OpenCVForUnity.Rect;

namespace RealTimeFaceRecognitionSample
{

    /// <summary>
    /// Easily detect objects such as faces or eyes (using LBP or Haar Cascades).
    /// Code is the rewrite of https://github.com/MasteringOpenCV/code/tree/master/Chapter8_FaceRecognition using the “OpenCV for Unity”.
    /// </summary>
    public static class DetectObject
    {
        private static Mat gray;
        private static Mat inputImg;
        private static Mat equalizedImg;
        private static MatOfRect matOfRectObjects = new MatOfRect ();

        // Search for objects such as faces in the image using the given parameters, storing the multiple cv::Rects into 'objects'.
        // Can use Haar cascades or LBP cascades for Face Detection, or even eye, mouth, or car detection.
        // Input is temporarily shrunk to 'scaledWidth' for much faster detection, since 200 is enough to find faces.
        private static void detectObjectsCustom (Mat img, CascadeClassifier cascade, out List<Rect> objects, int scaledWidth, int flags, Size minFeatureSize, float searchScaleFactor, int minNeighbors)
        {
            
            // If the input image is not grayscale, then convert the RGB or RGBA color image to grayscale.
            if (gray == null)
                gray = new Mat ();

            if (img.channels () == 3) {
                Imgproc.cvtColor (img, gray, Imgproc.COLOR_RGB2GRAY);
            } else if (img.channels () == 4) {
                Imgproc.cvtColor (img, gray, Imgproc.COLOR_RGBA2GRAY);
            } else {
                // Access the input image directly, since it is already grayscale.
                gray = img;
            }

            // Possibly shrink the image, to run much faster.
            if (inputImg == null)
                inputImg = new Mat ();

            float scale = img.cols () / (float)scaledWidth;
            if (img.cols () > scaledWidth) {
                // Shrink the image while keeping the same aspect ratio.
                int scaledHeight = (int)Mathf.Round (img.rows () / scale);
                Imgproc.resize (gray, inputImg, new Size (scaledWidth, scaledHeight));
            } else {
                // Access the input image directly, since it is already small.
                inputImg = gray;
            }

            // Standardize the brightness and contrast to improve dark images.
            if (equalizedImg == null)
                equalizedImg = new Mat ();

            Imgproc.equalizeHist (inputImg, equalizedImg);

            // Detect objects in the small grayscale image.
            cascade.detectMultiScale (equalizedImg, matOfRectObjects, searchScaleFactor, minNeighbors, flags, minFeatureSize, new Size ());
            objects = matOfRectObjects.toList ();

            // Enlarge the results if the image was temporarily shrunk before detection.
            if (img.cols () > scaledWidth) {
                for (int i = 0; i < objects.Count; i++) {
                    objects [i].x = (int)Mathf.Round (objects [i].x * scale);
                    objects [i].y = (int)Mathf.Round (objects [i].y * scale);
                    objects [i].width = (int)Mathf.Round (objects [i].width * scale);
                    objects [i].height = (int)Mathf.Round (objects [i].height * scale);
                }
            }

            // Make sure the object is completely within the image, in case it was on a border.
            for (int i = 0; i < objects.Count; i++) {
                if (objects [i].x < 0)
                    objects [i].x = 0;
                if (objects [i].y < 0)
                    objects [i].y = 0;
                if (objects [i].x + objects [i].width > img.cols ())
                    objects [i].x = img.cols () - objects [i].width;
                if (objects [i].y + objects [i].height > img.rows ())
                    objects [i].y = img.rows () - objects [i].height;
            }
            // Return with the detected face rectangles stored in "objects".
        }

        // Search for just a single object in the image, such as the largest face, storing the result into 'largestObject'.
        // Can use Haar cascades or LBP cascades for Face Detection, or even eye, mouth, or car detection.
        // Input is temporarily shrunk to 'scaledWidth' for much faster detection, since 200 is enough to find faces.
        // Note: detectLargestObject() should be faster than detectManyObjects().
        public static void DetectLargestObject (Mat img, CascadeClassifier cascade, out Rect largestObject, int scaledWidth = 320)
        {
            // Only search for just 1 object (the biggest in the image).
            int flags = Objdetect.CASCADE_FIND_BIGGEST_OBJECT;// | CASCADE_DO_ROUGH_SEARCH;
            // Smallest object size.
            Size minFeatureSize = new Size (20, 20);
            // How detailed should the search be. Must be larger than 1.0.
            float searchScaleFactor = 1.1f;
            // How much the detections should be filtered out. This should depend on how bad false detections are to your system.
            // minNeighbors=2 means lots of good+bad detections, and minNeighbors=6 means only good detections are given but some are missed.
            int minNeighbors = 4;

            // Perform Object or Face Detection, looking for just 1 object (the biggest in the image).
            List<Rect> objects;
            detectObjectsCustom (img, cascade, out objects, scaledWidth, flags, minFeatureSize, searchScaleFactor, minNeighbors);
            if (objects.Count > 0) {
                // Return the only detected object.
                largestObject = (Rect)objects [0];
            } else {
                // Return an invalid rect.
                largestObject = new Rect (-1, -1, -1, -1);
            }
        }

        // Search for many objects in the image, such as all the faces, storing the results into 'objects'.
        // Can use Haar cascades or LBP cascades for Face Detection, or even eye, mouth, or car detection.
        // Input is temporarily shrunk to 'scaledWidth' for much faster detection, since 200 is enough to find faces.
        // Note: detectLargestObject() should be faster than detectManyObjects().
        public static void DetectManyObjects (Mat img, CascadeClassifier cascade, out List<Rect> objects, int scaledWidth = 320)
        {
            // Search for many objects in the one image.
            int flags = Objdetect.CASCADE_SCALE_IMAGE;

            // Smallest object size.
            Size minFeatureSize = new Size (20, 20);
            // How detailed should the search be. Must be larger than 1.0.
            float searchScaleFactor = 1.1f;
            // How much the detections should be filtered out. This should depend on how bad false detections are to your system.
            // minNeighbors=2 means lots of good+bad detections, and minNeighbors=6 means only good detections are given but some are missed.
            int minNeighbors = 4;

            // Perform Object or Face Detection, looking for many objects in the one image.
            detectObjectsCustom (img, cascade, out objects, scaledWidth, flags, minFeatureSize, searchScaleFactor, minNeighbors);
        }
    }
}