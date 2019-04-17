using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class fps : MonoBehaviour {

    public int Framerate;



    private void Awake()
    {
        //Screen.SetResolution(1280, 720, false);
        //QualitySettings.vSyncCount = 0;
        //QualitySettings.SetQualityLevel(0);
        //Application.targetFrameRate = 300;
    }
    // Use this for initialization
    void Start () {
        //StartCoroutine(changeFramerate());

    }

    // Update is called once per frame
    void Update () {
        Framerate = (int)(1f / Time.unscaledDeltaTime);

    }

    IEnumerator changeFramerate()
    {
        yield return new WaitForSeconds(1);
        Application.targetFrameRate = 30;
    }

}
