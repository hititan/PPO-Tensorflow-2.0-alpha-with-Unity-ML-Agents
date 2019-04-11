using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using MLAgents;

public class TF20Academy : Academy
{
    private float myTimeScale = 1;

    public override void AcademyReset()
    {
        Time.timeScale = myTimeScale;
        this.SetIsInference(true);
    }

    public override void AcademyStep()
    {
        Time.timeScale = myTimeScale;
        
    }

    void OnGUI()
    {

        if (GUI.Button(new Rect(10, 10, 181, 25), "NORMAL - SPEED"))
        {
            myTimeScale = 1f;
            this.SetIsInference(true);
        }

        if (GUI.Button(new Rect(10, 40, 181, 25), "20 x SPEED"))
        {
            myTimeScale = 20f;
            this.SetIsInference(true);
        }

        if (GUI.Button(new Rect(10, 70, 181, 25), "100 x SPEED"))
        {
            myTimeScale = 100f;
            this.SetIsInference(true);

        }

    }
}
