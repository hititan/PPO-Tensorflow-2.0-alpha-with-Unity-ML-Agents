//using System;
using UnityEngine;
using MLAgents;


public class SorterArmAgent : Agent
{
    public GameObject Ball;
    public float speed = 360; // bei 0.01 s fixed time step = 1 s for 360 degrees
    public Vector3 rotation;

    private string agentName;

    public override void InitializeAgent()
    {
        // Set gravity for cm/sec**2  
        Physics.gravity = new Vector3(0, -981.0f, 0);
        agentName = this.gameObject.name;
    }

    public override void CollectObservations()
    {
        foreach (int co in System.Enum.GetValues(typeof(sorterEnums.ballColors)))
        {
            AddVectorObs((int)Ball.GetComponent<Ball>().GetMyColor() == co ? 1.0f : 0.0f);
        }

        AddVectorObs(this.transform.rotation.eulerAngles.y / 360);
        //AddVectorObs(LiftingArm.transform.localPosition.y / 10);

    }

    public override void AgentAction(float[] act, string textAction)
    {
        int action = Mathf.FloorToInt(act[0]);

        switch (action)
        {
            case 1:
                this.transform.Rotate(Vector3.up * Time.fixedDeltaTime * speed);
                break;

            case 2:
                this.transform.Rotate(Vector3.down * Time.fixedDeltaTime * speed);
                break;

            default:
                //case 0: print("Default action...");
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
                //SetReward(-1f);
                Monitor.Log("Reward for " + agentName, GetReward());
                Done();
            }
    }

    public override void AgentReset()
    {
       
        Ball.GetComponent<Ball>().reset();
        this.transform.rotation = new Quaternion();
        this.transform.Rotate(Vector3.up, Random.Range(-90, 90));
    }

    public override void AgentOnDone()
    {
    }


}

