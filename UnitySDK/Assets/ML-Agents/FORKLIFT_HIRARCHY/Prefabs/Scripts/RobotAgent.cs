using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using MLAgents;

public class RobotAgent : Agent
{
    [Header("Wheel Settings")]
    public GameObject leftWheel;
    public GameObject rightWheel;
    Rigidbody rbLeftWheel;
    Rigidbody rbRightWheel;

    [Header("Fork Settings")]
    public GameObject forks;
    public float forkMin = -0.2f;
    public float forkMax = 1f;
    float position;

    [Header("Speed Settings")]
    public float maxAngularVelocity = 100;
    const float radians = Mathf.PI / 180f;
    public float rpm = 1;
    float wheelRPM;

    private string sceneName;

    public override void InitializeAgent()
    {
        sceneName = UnityEngine.SceneManagement.SceneManager.GetActiveScene().name;
        print("Scene Name: " + sceneName);

        rbLeftWheel = leftWheel.GetComponent<Rigidbody>();
        rbLeftWheel.maxAngularVelocity = maxAngularVelocity;
        rbRightWheel = rightWheel.GetComponent<Rigidbody>();
        rbRightWheel.maxAngularVelocity = maxAngularVelocity;

        wheelRPM = radians * 360f * rpm;
        position = forkMin;
    }

    public override void CollectObservations()
    {

        AddVectorObs(this.transform.position);
        AddVectorObs(this.transform.rotation.eulerAngles);
        AddVectorObs(0f);
        AddVectorObs(0f);
        AddVectorObs(0f);
        AddVectorObs(0f);
        

    }

    public override void AgentAction(float[] vectorAction, string textAction)
    {
        var act = (int)vectorAction[0];



        switch (act)
        {
            case 1:
                //Left
                rbLeftWheel.angularVelocity = rbLeftWheel.transform.up * wheelRPM;
                rbRightWheel.angularVelocity = -rbRightWheel.transform.up * wheelRPM;
                break;
            case 2:
                //Right
                rbLeftWheel.angularVelocity = -rbLeftWheel.transform.up * wheelRPM;
                rbRightWheel.angularVelocity = rbRightWheel.transform.up * wheelRPM; break;
            case 3:
                // Forward
                rbLeftWheel.angularVelocity = -rbLeftWheel.transform.up * wheelRPM;
                rbRightWheel.angularVelocity = -rbRightWheel.transform.up * wheelRPM;
                break;
            case 4:
                // Back
                rbLeftWheel.angularVelocity = rbLeftWheel.transform.up * wheelRPM;
                rbRightWheel.angularVelocity = rbRightWheel.transform.up * wheelRPM;
                break;
            case 5:
                if (position < forkMax)
                {
                    position = position + 0.01f;
                    forks.transform.localPosition = new Vector3(0, position, 0);
                }
                break;
            case 6:
                if (position > forkMin)
                {
                    position = position - 0.01f;
                    forks.transform.localPosition = new Vector3(0, position, 0);
                }
                break;
            default:
                //print("default");
                rbLeftWheel.angularVelocity = rbLeftWheel.transform.up * 0f;
                rbRightWheel.angularVelocity = rbRightWheel.transform.up * 0f;
                break;
        }

        checkForRewards();
    }

    void checkForRewards()
    {


    }


    public override void AgentReset()
    {

    }

}
