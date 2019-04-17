using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using MLAgents;

public class LiftingArmAgent : Agent
{
    public float speed = 20;
    public float movement;
    public float upperLimit = 7.0f;
    public float lowerLimit = 4.0f;
    public Vector3 resetPosition;

    public GameObject Ball;

    private string agentName;


    // Use this for initialization
    public override void InitializeAgent()
    {
        resetPosition = this.transform.localPosition;

        this.transform.localPosition = new Vector3(transform.localPosition.x,
                                                 Mathf.Clamp(this.transform.localPosition.y, lowerLimit, upperLimit),
                                                 transform.localPosition.z);

        movement = Time.fixedDeltaTime * speed;

        agentName = this.gameObject.name;
    }

    public override void CollectObservations()
    {
        foreach (int co in System.Enum.GetValues(typeof(sorterEnums.ballColors)))
        {
            AddVectorObs((int)Ball.GetComponent<Ball>().GetMyColor() == co ? 1.0f : 0.0f);
        }

        //AddVectorObs(this.transform.rotation.eulerAngles.y / 360);
        AddVectorObs(this.transform.localPosition.y / 10);

    }

    public override void AgentAction(float[] act, string textAction)
    {
        int action = Mathf.FloorToInt(act[0]);

        switch (action)
        {
            case 1:
                moveUp();
                break;

            case 2:
                moveDown();
                break;

            default:
                //print("Default action...");
                break;
        }


        if (Ball.GetComponent<Ball>().giveReward)
        {
            SetReward(1);
            Monitor.Log("Reward for " + agentName, GetReward());
            Done();
        }
        else if (Ball.GetComponent<Ball>().stopFlag)
        {
            //SetReward(-1);
            Monitor.Log("Reward for " + agentName, GetReward());
            Done();
        }
    }

    public void moveUp()
    {
        if (this.transform.localPosition.y <= upperLimit - 0.1)
        {
            this.transform.Translate(Vector3.up * movement);
        }
    }

    public void moveDown()
    {
        if (this.transform.localPosition.y >= lowerLimit + 0.1)
        {
            this.transform.Translate(Vector3.down * movement);
        }
    }

    public override void AgentReset()
    {
        this.transform.localPosition = resetPosition;
    }


}
