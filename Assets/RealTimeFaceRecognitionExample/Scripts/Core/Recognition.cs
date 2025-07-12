using OpenCVForUnity.CoreModule;
using OpenCVForUnity.FaceModule;
using System.Collections.Generic;
using UnityEngine;

namespace RealTimeFaceRecognitionExample
{
    /// <summary>
    /// Train the face recognition system on a given dataset, and recognize the person from a given image.
    /// The code is a rewrite of https://github.com/MasteringOpenCV/code/tree/master/Chapter8_FaceRecognition using "OpenCV for Unity".
    /// </summary>
    public static class Recognition
    {
        // Start training from the collected faces.
        // The face recognition algorithm can be one of these and perhaps more, depending on your version of OpenCV, which must be atleast v2.4.1:
        //    "FaceRecognizer.Eigenfaces":  Eigenfaces, also referred to as PCA (Turk and Pentland, 1991).
        //    "FaceRecognizer.Fisherfaces": Fisherfaces, also referred to as LDA (Belhumeur et al, 1997).
        //    "FaceRecognizer.LBPH":        Local Binary Pattern Histograms (Ahonen et al, 2006).
        public static BasicFaceRecognizer LearnCollectedFaces(List<Mat> preprocessedFaces, List<int> faceLabels, string facerecAlgorithm = "FaceRecognizer.Eigenfaces")
        {
            BasicFaceRecognizer model = null;

            Debug.Log("Learning the collected faces using the [" + facerecAlgorithm + "] algorithm ...");

            if (facerecAlgorithm == "FaceRecognizer.Fisherfaces")
            {
                model = FisherFaceRecognizer.create();
            }
            else if (facerecAlgorithm == "FaceRecognizer.Eigenfaces")
            {
                model = EigenFaceRecognizer.create();
            }

            if (model == null)
            {
                Debug.LogError("ERROR: The FaceRecognizer algorithm [" + facerecAlgorithm + "] is not available in your version of OpenCV. Please update to OpenCV v2.4.1 or newer.");
                //exit(1);
            }

            // Do the actual training from the collected faces. Might take several seconds or minutes depending on input!
            MatOfInt labels = new MatOfInt();
            labels.fromList(faceLabels);
            model.train(preprocessedFaces, labels);

            return model;
        }

        // Convert the matrix row or column (float matrix) to a rectangular 8-bit image that can be displayed or saved.
        // Scales the values to be between 0 to 255.
        private static Mat getImageFrom1DFloatMat(Mat matrixRow, int height)
        {
            // Make it a rectangular shaped image instead of a single row.
            Mat rectangularMat = matrixRow.reshape(1, height);
            // Scale the values to be between 0 to 255 and store them as a regular 8-bit uchar image.
            Mat dst = new Mat();
            Core.normalize(rectangularMat, dst, 0, 255, Core.NORM_MINMAX, CvType.CV_8UC1);
            return dst;
        }

        // Show the internal face recognition data, to help debugging.
        public static void ShowTrainingDebugData(BasicFaceRecognizer model, int faceWidth, int faceHeight)
        {
            // TODO...
        }

        // Generate an approximately reconstructed face by back-projecting the eigenvectors & eigenvalues of the given (preprocessed) face.
        public static Mat ReconstructFace(BasicFaceRecognizer model, Mat preprocessedFace)
        {
            // Since we can only reconstruct the face for some types of FaceRecognizer models (ie: Eigenfaces or Fisherfaces),
            // we should surround the OpenCV calls by a try/catch block so we don't crash for other models.
            try
            {

                // Get some required data from the FaceRecognizer model.
                Mat eigenvectors = model.getEigenVectors();
                Mat averageFaceRow = model.getMean();

                int faceHeight = preprocessedFace.rows();

                // Project the input image onto the PCA subspace.
                Mat projection = subspaceProject(eigenvectors, averageFaceRow, preprocessedFace.reshape(1, 1));
                //printMatInfo(projection, "projection");

                // Generate the reconstructed face back from the PCA subspace.
                Mat reconstructionRow = subspaceReconstruct(eigenvectors, averageFaceRow, projection);
                //printMatInfo(reconstructionRow, "reconstructionRow");

                // Convert the float row matrix to a regular 8-bit image. Note that we
                // shouldn't use "getImageFrom1DFloatMat()" because we don't want to normalize
                // the data since it is already at the perfect scale.

                // Make it a rectangular shaped image instead of a single row.
                Mat reconstructionMat = reconstructionRow.reshape(1, faceHeight);
                // Convert the floating-point pixels to regular 8-bit uchar pixels.
                Mat reconstructedFace = new Mat(reconstructionMat.size(), CvType.CV_8UC1);
                reconstructionMat.convertTo(reconstructedFace, CvType.CV_8UC1, 1, 0);
                //printMatInfo(reconstructedFace, "reconstructedFace");

                return reconstructedFace;

            }
            catch (CvException e)
            {
                Debug.Log("WARNING: Missing FaceRecognizer properties." + e);
                return new Mat();
            }
        }

        // Compare two images by getting the L2 error (square-root of sum of squared error).
        public static double GetSimilarity(Mat A, Mat B)
        {
            if (A.rows() > 0 && A.rows() == B.rows() && A.cols() > 0 && A.cols() == B.cols())
            {
                // Calculate the L2 relative error between the 2 images.
                double errorL2 = Core.norm(A, B, Core.NORM_L2);
                // Convert to a reasonable scale, since L2 error is summed across all pixels of the image.
                double similarity = errorL2 / (double)(A.rows() * A.cols());
                return similarity;
            }
            else
            {
                //Debug.Log("WARNING: Images have a different size in 'GetSimilarity()'.");
                return 100000000.0;  // Return a bad value
            }
        }

        // Porting the cv::subspaceProject() function from c++ opencv.
        // https://github.com/opencv/opencv/blob/master/modules/core/src/lda.cpp
        private static Mat subspaceProject(Mat W, Mat mean, Mat src)
        {
            // get number of samples and dimension
            int n = src.rows();
            int d = src.cols();
            // make sure the data has the correct shape
            if (W.rows() != d)
            {
                string error_message = string.Format("Wrong shapes for given matrices. Was size(src) = ({0},{1}), size(W) = ({2},{3}).", src.rows(), src.cols(), W.rows(), W.cols());
                throw new CvException(error_message);
            }
            // make sure mean is correct if not empty
            if (mean.total() > 0 && (mean.total() != d))
            {
                string error_message = string.Format("Wrong mean shape for the given data matrix. Expected {0}, but was {1}.", d, mean.total());
                throw new CvException(error_message);
            }
            // create temporary matrices
            Mat X = new Mat(), Y = new Mat();
            // make sure you operate on correct type
            src.convertTo(X, W.type());
            // safe to do, because of above assertion
            if (mean.total() > 0)
            {
                for (int i = 0; i < n; i++)
                {
                    Mat r_i = X.row(i);
                    Core.subtract(r_i, mean.reshape(1, 1), r_i);
                }
            }
            // finally calculate projection as Y = (X-mean)*W
            Core.gemm(X, W, 1.0, new Mat(), 0.0, Y);
            return Y;
        }

        // Porting the cv::subspaceReconstruct() function from c++ opencv.
        // https://github.com/opencv/opencv/blob/master/modules/core/src/lda.cpp
        private static Mat subspaceReconstruct(Mat W, Mat mean, Mat src)
        {
            // get number of samples and dimension
            int n = src.rows();
            int d = src.cols();
            // make sure the data has the correct shape
            if (W.cols() != d)
            {
                string error_message = string.Format("Wrong shapes for given matrices. Was size(src) = ({0},{1}), size(W) = ({2},{3}).", src.rows(), src.cols(), W.rows(), W.cols());
                throw new CvException(error_message);
            }
            // make sure mean is correct if not empty
            if (mean.total() > 0 && (mean.total() != W.rows()))
            {
                string error_message = string.Format("Wrong mean shape for the given eigenvector matrix. Expected {0}, but was {1}.", W.cols(), mean.total());
                throw new CvException(error_message);
            }
            // initialize temporary matrices
            Mat X = new Mat(), Y = new Mat();
            // copy data & make sure we are using the correct type
            src.convertTo(Y, W.type());
            // calculate the reconstruction
            Core.gemm(Y, W, 1.0, new Mat(), 0.0, X, Core.GEMM_2_T);
            // safe to do because of above assertion
            if (mean.total() > 0)
            {
                for (int i = 0; i < n; i++)
                {
                    Mat r_i = X.row(i);
                    Core.add(r_i, mean.reshape(1, 1), r_i);
                }
            }
            return X;
        }
    }
}