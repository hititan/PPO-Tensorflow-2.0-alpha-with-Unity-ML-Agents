using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using MLAgents;

public class CartPole_Academy : Academy
{

    private float myTimeScale = 1;

    public override void AcademyReset()
    {

        Time.timeScale = myTimeScale;
    }

    public override void AcademyStep()
    {

        Time.timeScale = myTimeScale;
    }

    void OnGUI()
    {

        if (GUI.Button(new Rect(10, 10, 200, 25), "SPEED"))
        {
            if (myTimeScale == 1)
            {
                myTimeScale = 20f;
            }
            else
            {
                myTimeScale = 1f;
            }

            // if (GUI.Button(new Rect(10, 10, 50, 30), "Speed"))
            // {
            //     if (this.GetIsInference())
            //     {
            //         this.SetIsInference(false);
            //     }
            //     else
            //     {
            //         this.SetIsInference(true);
            //     }
            // }

        }

    }
}
