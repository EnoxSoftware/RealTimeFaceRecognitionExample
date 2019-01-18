using System;
using UnityEngine;
using OpenCVForUnity.CoreModule;
using OpenCVForUnity.ObjdetectModule;
using OpenCVForUnity.ImgprocModule;
using OpenCVForUnity.UnityUtils;
using Rect = OpenCVForUnity.CoreModule.Rect;

namespace RealTimeFaceRecognitionExample
{
    /// <summary>
    /// Easily preprocess face images, for face recognition.
    /// This code is a rewrite of https://github.com/MasteringOpenCV/code/tree/master/Chapter8_FaceRecognition using "OpenCV for Unity".
    /// </summary>
    public static class PreprocessFace
    {
        static Scalar BLACK = new Scalar (0);
        static Scalar WHITE = new Scalar (255);
        static Scalar GRAY = new Scalar (128);

        const double DESIRED_LEFT_EYE_X = 0.16d;
        // Controls how much of the face is visible after preprocessing.
        const double DESIRED_LEFT_EYE_Y = 0.14d;
        const double FACE_ELLIPSE_CY = 0.40d;
        const double FACE_ELLIPSE_W = 0.50d;
        // Should be atleast 0.5
        const double FACE_ELLIPSE_H = 0.80d;
        // Controls how tall the face mask is.

        // Search for both eyes within the given face image. Returns the eye centers in 'leftEye' and 'rightEye',
        // or sets them to (-1,-1) if each eye was not found. Note that you can pass a 2nd eyeCascade if you
        // want to search eyes using 2 different cascades. For example, you could use a regular eye detector
        // as well as an eyeglasses detector, or a left eye detector as well as a right eye detector.
        // Or if you don't want a 2nd eye detection, just pass an uninitialized CascadeClassifier.
        // Can also store the searched left & right eye regions if desired.
        private static void detectBothEyes (Mat face, CascadeClassifier eyeCascade1, CascadeClassifier eyeCascade2, out Point leftEye, out Point rightEye, ref Rect searchedLeftEye, ref Rect searchedRightEye)
        {
            // Skip the borders of the face, since it is usually just hair and ears, that we don't care about.
            /*
                // For "2splits.xml": Finds both eyes in roughly 60% of detected faces, also detects closed eyes.
                const float EYE_SX = 0.12f;
                const float EYE_SY = 0.17f;
                const float EYE_SW = 0.37f;
                const float EYE_SH = 0.36f;
            */
            /*
                // For mcs.xml: Finds both eyes in roughly 80% of detected faces, also detects closed eyes.
                const float EYE_SX = 0.10f;
                const float EYE_SY = 0.19f;
                const float EYE_SW = 0.40f;
                const float EYE_SH = 0.36f;
            */

            // For default eye.xml or eyeglasses.xml: Finds both eyes in roughly 40% of detected faces, but does not detect closed eyes.
            const float EYE_SX = 0.16f;
            const float EYE_SY = 0.26f;
            const float EYE_SW = 0.30f;
            const float EYE_SH = 0.28f;

            int leftX = (int)Mathf.Round (face.cols () * EYE_SX);
            int topY = (int)Mathf.Round (face.rows () * EYE_SY);
            int widthX = (int)Mathf.Round (face.cols () * EYE_SW);
            int heightY = (int)Mathf.Round (face.rows () * EYE_SH);
            int rightX = (int)(Mathf.Round (face.cols () * (1f - EYE_SX - EYE_SW)));  // Start of right-eye corner

            Mat topLeftOfFace = new Mat (face, new Rect (leftX, topY, widthX, heightY));
            Mat topRightOfFace = new Mat (face, new Rect (rightX, topY, widthX, heightY));
            Rect leftEyeRect, rightEyeRect;

            // Return the search windows to the caller, if desired.
            if (searchedLeftEye != null)
                searchedLeftEye = new Rect (leftX, topY, widthX, heightY);
            if (searchedRightEye != null)
                searchedRightEye = new Rect (rightX, topY, widthX, heightY);

            // Search the left region, then the right region using the 1st eye detector.
            DetectObject.DetectLargestObject (topLeftOfFace, eyeCascade1, out leftEyeRect, topLeftOfFace.cols ());
            DetectObject.DetectLargestObject (topRightOfFace, eyeCascade1, out rightEyeRect, topRightOfFace.cols ());

            // If the eye was not detected, try a different cascade classifier.
            if (leftEyeRect.width <= 0 /*&& !eyeCascade2.empty()*/) {
                DetectObject.DetectLargestObject (topLeftOfFace, eyeCascade2, out leftEyeRect, topLeftOfFace.cols ());
                //if (leftEyeRect.width > 0)
                //    cout << "2nd eye detector LEFT SUCCESS" << endl;
                //else
                //    cout << "2nd eye detector LEFT failed" << endl;
            }
            //else
            //    cout << "1st eye detector LEFT SUCCESS" << endl;

            // If the eye was not detected, try a different cascade classifier.
            if (rightEyeRect.width <= 0 /*&& !eyeCascade2.empty ()*/) {
                DetectObject.DetectLargestObject (topRightOfFace, eyeCascade2, out rightEyeRect, topRightOfFace.cols ());
                //if (rightEyeRect.width > 0)
                //    cout << "2nd eye detector RIGHT SUCCESS" << endl;
                //else
                //    cout << "2nd eye detector RIGHT failed" << endl;
            }
            //else
            //    cout << "1st eye detector RIGHT SUCCESS" << endl;

            if (leftEyeRect.width > 0) {   // Check if the eye was detected.
                leftEyeRect.x += leftX;    // Adjust the left-eye rectangle because the face border was removed.
                leftEyeRect.y += topY;
                leftEye = new Point (leftEyeRect.x + leftEyeRect.width / 2, leftEyeRect.y + leftEyeRect.height / 2);
            } else {
                leftEye = new Point (-1, -1);    // Return an invalid point
            }

            if (rightEyeRect.width > 0) {   // Check if the eye was detected.
                rightEyeRect.x += rightX; // Adjust the right-eye rectangle, since it starts on the right side of the image.
                rightEyeRect.y += topY;  // Adjust the right-eye rectangle because the face border was removed.
                rightEye = new Point (rightEyeRect.x + rightEyeRect.width / 2, rightEyeRect.y + rightEyeRect.height / 2);
            } else {
                rightEye = new Point (-1, -1);    // Return an invalid point
            }
        }

        // Histogram Equalize seperately for the left and right sides of the face.
        private static void equalizeLeftAndRightHalves (Mat faceImg)
        {
            // It is common that there is stronger light from one half of the face than the other. In that case,
            // if you simply did histogram equalization on the whole face then it would make one half dark and
            // one half bright. So we will do histogram equalization separately on each face half, so they will
            // both look similar on average. But this would cause a sharp edge in the middle of the face, because
            // the left half and right half would be suddenly different. So we also histogram equalize the whole
            // image, and in the middle part we blend the 3 images together for a smooth brightness transition.

            int w = faceImg.cols ();
            int h = faceImg.rows ();

            // 1) First, equalize the whole face.
            using (Mat wholeFace = new Mat (h, w, CvType.CV_8UC1)) {
                Imgproc.equalizeHist (faceImg, wholeFace);

                // 2) Equalize the left half and the right half of the face separately.
                int midX = w / 2;
                using (Mat leftSide = new Mat (faceImg, new Rect (0, 0, midX, h)))
                using (Mat rightSide = new Mat (faceImg, new Rect (midX, 0, w - midX, h))) {
                    Imgproc.equalizeHist (leftSide, leftSide);
                    Imgproc.equalizeHist (rightSide, rightSide);

                    // 3) Combine the left half and right half and whole face together, so that it has a smooth transition.
                    byte[] wholeFace_byte = new byte[wholeFace.total () * wholeFace.elemSize ()];
                    Utils.copyFromMat<byte> (wholeFace, wholeFace_byte);
                    byte[] leftSide_byte = new byte[leftSide.total () * leftSide.elemSize ()];
                    Utils.copyFromMat<byte> (leftSide, leftSide_byte);
                    byte[] rightSide_byte = new byte[rightSide.total () * rightSide.elemSize ()];
                    Utils.copyFromMat<byte> (rightSide, rightSide_byte);

                    int leftSide_w = leftSide.cols ();
                    int rightSide_w = rightSide.cols ();

                    for (int y = 0; y < h; y++) {
                        for (int x = 0; x < w; x++) {
                            byte wv = wholeFace_byte [y * w + x];
                            if (x < w / 4) {   // Left 25%: just use the left face.
                                wv = leftSide_byte [y * leftSide_w + x];
                            } else if (x < w * 2 / 4) {   // Mid-left 25%: blend the left face & whole face.
                                byte lv = leftSide_byte [y * leftSide_w + x];

                                // Blend more of the whole face as it moves further right along the face.
                                float f = (x - w * 1 / 4) / (w * 0.25f);
                                wv = (byte)Mathf.Round ((1.0f - f) * lv + f * wv);
                            } else if (x < w * 3 / 4) {   // Mid-right 25%: blend the right face & whole face.
                                byte rv = rightSide_byte [y * rightSide_w + x - midX];

                                // Blend more of the right-side face as it moves further right along the face.
                                float f = (x - w * 2 / 4) / (w * 0.25f);
                                wv = (byte)Mathf.Round ((1.0f - f) * wv + f * rv);
                            } else {   // Right 25%: just use the right face.
                                wv = rightSide_byte [y * rightSide_w + x - midX];
                            }
                        }// end x loop
                    }//end y loop
                    Utils.copyToMat (wholeFace_byte, faceImg);
                }
            }
        }

        // Create a grayscale face image that has a standard size and contrast & brightness.
        // "srcImg" should be a copy of the whole color camera frame, so that it can draw the eye positions onto.
        // If 'doLeftAndRightSeparately' is true, it will process left & right sides seperately,
        // so that if there is a strong light on one side but not the other, it will still look OK.
        // Performs Face Preprocessing as a combination of:
        //  - geometrical scaling, rotation and translation using Eye Detection,
        //  - smoothing away image noise using a Bilateral Filter,
        //  - standardize the brightness on both left and right sides of the face independently using separated Histogram Equalization,
        //  - removal of background and hair using an Elliptical Mask.
        // Returns either a preprocessed face square image or NULL (ie: couldn't detect the face and 2 eyes).
        // If a face is found, it can store the rect coordinates into 'storeFaceRect' and 'storeLeftEye' & 'storeRightEye' if given,
        // and eye search regions into 'searchedLeftEye' & 'searchedRightEye' if given.
        public static Mat GetPreprocessedFace (Mat srcImg, int desiredFaceWidth, CascadeClassifier faceCascade, CascadeClassifier eyeCascade1, CascadeClassifier eyeCascade2, bool doLeftAndRightSeparately, ref Rect storeFaceRect, ref Point storeLeftEye, ref Point storeRightEye, ref Rect searchedLeftEye, ref Rect searchedRightEye)
        {
            // Use square faces.
            int desiredFaceHeight = desiredFaceWidth;

            // Mark the detected face region and eye search regions as invalid, in case they aren't detected.
            if (storeFaceRect != null)
                storeFaceRect.width = -1;
            if (storeLeftEye != null)
                storeLeftEye.x = -1;
            if (storeRightEye != null)
                storeRightEye.x = -1;
            if (searchedLeftEye != null)
                searchedLeftEye.width = -1;
            if (searchedRightEye != null)
                searchedRightEye.width = -1;

            // Find the largest face.
            Rect faceRect;
            DetectObject.DetectLargestObject (srcImg, faceCascade, out faceRect);

            // Check if a face was detected.
            if (faceRect.width > 0) {

                // Give the face rect to the caller if desired.
                if (storeFaceRect != null)
                    storeFaceRect = faceRect;

                // Get the detected face image.
                using (Mat faceImg = new Mat (srcImg, faceRect)) {

                    // If the input image is not grayscale, then convert the RGB or RGBA color image to grayscale.
                    using (Mat gray = new Mat ()) {
                        if (faceImg.channels () == 3) {
                            Imgproc.cvtColor (faceImg, gray, Imgproc.COLOR_RGB2GRAY);
                        } else if (faceImg.channels () == 4) {
                            Imgproc.cvtColor (faceImg, gray, Imgproc.COLOR_RGBA2GRAY);
                        } else {
                            // Access the input image directly, since it is already grayscale.
                            faceImg.copyTo (gray);
                        }

                        // Search for the 2 eyes at the full resolution, since eye detection needs max resolution possible!
                        Point leftEye, rightEye;
                        detectBothEyes (gray, eyeCascade1, eyeCascade2, out leftEye, out rightEye, ref searchedLeftEye, ref searchedRightEye);

                        // Give the eye results to the caller if desired.
                        if (storeLeftEye != null)
                            storeLeftEye = leftEye;
                        if (storeRightEye != null)
                            storeRightEye = rightEye;

                        // Check if both eyes were detected.
                        if (leftEye.x >= 0 && rightEye.x >= 0) {

                            // Make the face image the same size as the training images.

                            // Since we found both eyes, lets rotate & scale & translate the face so that the 2 eyes
                            // line up perfectly with ideal eye positions. This makes sure that eyes will be horizontal,
                            // and not too far left or right of the face, etc.

                            // Get the center between the 2 eyes.
                            Point eyesCenter = new Point ((leftEye.x + rightEye.x) * 0.5f, (leftEye.y + rightEye.y) * 0.5f);
                            // Get the angle between the 2 eyes.
                            double dy = (rightEye.y - leftEye.y);
                            double dx = (rightEye.x - leftEye.x);
                            double len = Math.Sqrt (dx * dx + dy * dy);
                            double angle = Math.Atan2 (dy, dx) * 180.0d / Math.PI; // Convert from radians to degrees.

                            // Hand measurements shown that the left eye center should ideally be at roughly (0.19, 0.14) of a scaled face image.
                            const double DESIRED_RIGHT_EYE_X = (1.0d - DESIRED_LEFT_EYE_X);
                            // Get the amount we need to scale the image to be the desired fixed size we want.
                            double desiredLen = (DESIRED_RIGHT_EYE_X - DESIRED_LEFT_EYE_X) * desiredFaceWidth;
                            double scale = desiredLen / len;
                            // Get the transformation matrix for rotating and scaling the face to the desired angle & size.
                            Mat rot_mat = Imgproc.getRotationMatrix2D (eyesCenter, angle, scale);
                            // Shift the center of the eyes to be the desired center between the eyes.
                            double[] shiftX = rot_mat.get (0, 2);
                            shiftX [0] += desiredFaceWidth * 0.5f - eyesCenter.x;
                            rot_mat.put (0, 2, shiftX);
                            double[] shiftY = rot_mat.get (1, 2);
                            shiftY [0] += desiredFaceHeight * DESIRED_LEFT_EYE_Y - eyesCenter.y;
                            rot_mat.put (1, 2, shiftY);

                            // Rotate and scale and translate the image to the desired angle & size & position!
                            // Note that we use 'w' for the height instead of 'h', because the input face has 1:1 aspect ratio.
                            using (Mat warped = new Mat (desiredFaceHeight, desiredFaceWidth, CvType.CV_8UC1, GRAY)) // Clear the output image to a default grey.
                            using (Mat filtered = new Mat (desiredFaceHeight, desiredFaceWidth, CvType.CV_8UC1))
                            using (Mat mask = new Mat (desiredFaceHeight, desiredFaceWidth, CvType.CV_8UC1, BLACK)) { // Start with an empty mask.
                                Imgproc.warpAffine (gray, warped, rot_mat, warped.size ());
                                //imshow("warped", warped);

                                // Give the image a standard brightness and contrast, in case it was too dark or had low contrast.
                                if (!doLeftAndRightSeparately) {
                                    // Do it on the whole face.
                                    Imgproc.equalizeHist (warped, warped);
                                } else {
                                    // Do it seperately for the left and right sides of the face.
                                    equalizeLeftAndRightHalves (warped);
                                }
                                //imshow("equalized", warped);

                                // Use the "Bilateral Filter" to reduce pixel noise by smoothing the image, but keeping the sharp edges in the face.
                                Imgproc.bilateralFilter (warped, filtered, 0, 20.0, 2.0);
                                //imshow("filtered", filtered);

                                // Filter out the corners of the face, since we mainly just care about the middle parts.
                                // Draw a filled ellipse in the middle of the face-sized image.
                                Point faceCenter = new Point (desiredFaceWidth / 2, Math.Round (desiredFaceHeight * FACE_ELLIPSE_CY));
                                Size size = new Size (Math.Round (desiredFaceWidth * FACE_ELLIPSE_W), Math.Round (desiredFaceHeight * FACE_ELLIPSE_H));
                                Imgproc.ellipse (mask, faceCenter, size, 0, 0, 360, WHITE, Core.FILLED);
                                //imshow("mask", mask);

                                // Use the mask, to remove outside pixels.
                                Mat dstImg = new Mat (desiredFaceHeight, desiredFaceWidth, CvType.CV_8UC1, GRAY); // Clear the output image to a default gray.
                                /*
                                namedWindow("filtered");
                                imshow("filtered", filtered);
                                namedWindow("dstImg");
                                imshow("dstImg", dstImg);
                                namedWindow("mask");
                                imshow("mask", mask);
                                */
                                // Apply the elliptical mask on the face.
                                filtered.copyTo (dstImg, mask);  // Copies non-masked pixels from filtered to dstImg.
                                //imshow("dstImg", dstImg);

                                return dstImg;
                            }
                        }
                    }
                }
                /*
                else {
                    // Since no eyes were found, just do a generic image resize.
                    resize(gray, tmpImg, Size(w,h));
                }
                */
            }
            return null;
        }
    }
}