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

    public GameObject MainBody;
    public GameObject Palette;

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
        if (brain.name == "RobotPlayerBrain")
        {
            for (int i=0 ; i < brain.brainParameters.vectorObservationSize; i++)
            {
                AddVectorObs(0f);
            }
        }
        else
        {
            AddVectorObs(Palette.transform.position.x / 10);
            AddVectorObs(Palette.transform.position.z / 10);
            AddVectorObs(MainBody.transform.position.x / 10);
            AddVectorObs(MainBody.transform.position.z / 10);
            AddVectorObs(MainBody.transform.rotation.eulerAngles.y * 3.14f / 180f);
            AddVectorObs(MainBody.GetComponent<Rigidbody>().velocity.x);
            AddVectorObs(MainBody.GetComponent<Rigidbody>().velocity.z);
            AddVectorObs(MainBody.GetComponent<Rigidbody>().angularVelocity.y);
        }
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
                SetReward(-0.1f);
                break;
            case 2:
                //Right
                rbLeftWheel.angularVelocity = -rbLeftWheel.transform.up * wheelRPM;
                rbRightWheel.angularVelocity = rbRightWheel.transform.up * wheelRPM; 
                SetReward(-0.1f);
                break;
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
        // checkForRewards();
    }

    void checkForRewards()
    {
        // Dense Rewards
        float distance = Vector3.Distance(MainBody.transform.position, Palette.transform.position);
        double reward = 10f / distance;
        reward = System.Math.Round(reward,0);
        
        if (reward == 1)
        {
            reward = 0; 
        }
        // print("Distance Reward: " + reward);
        SetReward(System.Convert.ToSingle(reward/10));
    }


    public override void AgentReset()
    {
        // Physics stability with setting all to zero positon and rotation
        // Set Physics time in Unity from 0.02 to 0.04
        // Set Physics Solver Iterations down to get more stable physics e.g. to 1

        MainBody.GetComponent<Rigidbody>().angularVelocity = Vector3.zero;
        MainBody.GetComponent<Rigidbody>().velocity = Vector3.zero;
        MainBody.transform.rotation = new Quaternion(0, 0, 0, 0);
        MainBody.transform.position = new Vector3(0, 0.4f, 0);

        Palette.transform.rotation = new Quaternion(0, 0, 0, 0);
        Palette.transform.position = new Vector3(Random.value * 14 - 7, 0.4f, Random.value * 14 - 7);
    }

    public void CollisionDetected(CollisionDetection colDect, Collider col)
    {
        // Collider with Trigger
        if (col.gameObject.name == "Palette")
        {
            // print("Collided with " + col.gameObject.name);
            SetReward(2f);
            Done();
        }

        if (col.gameObject.name == "Fence")
        {
            // print("Collided with " + col.gameObject.name);
            SetReward(-1f);
            Done();
        }
    }
}
