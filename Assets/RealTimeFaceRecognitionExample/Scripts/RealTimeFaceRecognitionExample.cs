using UnityEngine;
using System.Collections;

#if UNITY_5_3 || UNITY_5_3_OR_NEWER
using UnityEngine.SceneManagement;
#endif

namespace RealTimeFaceRecognitionExample
{
    public class RealTimeFaceRecognitionExample : MonoBehaviour
    {

        // Use this for initialization
        void Start ()
        {

        }

        // Update is called once per frame
        void Update ()
        {

        }

        public void OnShowLicenseButton ()
        {
            #if UNITY_5_3 || UNITY_5_3_OR_NEWER
			SceneManager.LoadScene ("ShowLicense");
            #else
            Application.LoadLevel ("ShowLicense");
            #endif
        }

        public void OnWebCamTextureRealTimeFaceRecognitionExample ()
        {
            #if UNITY_5_3 || UNITY_5_3_OR_NEWER
            SceneManager.LoadScene ("WebCamTextureRealTimeFaceRecognitionExample");
            #else
            Application.LoadLevel ("WebCamTextureRealTimeFaceRecognitionExample");
            #endif
        }
    }
}
