using System;
using System.Collections;
using UnityEngine;
using MLAgents;


public class BallSorterPlayer : Agent
{
    public GameObject Ball;
    public GameObject SorterArm;
    public GameObject LiftingArm;

    public bool useVisual= false;

    private string agentName;

    public override void InitializeAgent()
    {
        // Set gravity for cm/sec**2  
        Physics.gravity = new Vector3(0, -981.0f, 0);
        agentName = this.gameObject.name;

    }

    public override void CollectObservations()
    {
        if(!this.useVisual)
        {
            foreach (int co in System.Enum.GetValues(typeof(sorterEnums.ballColors)))
            {
                AddVectorObs((int)Ball.GetComponent<Ball>().GetMyColor() == co ? 1.0f : 0.0f);
            }

            AddVectorObs(SorterArm.transform.rotation.eulerAngles.y / 360);
            AddVectorObs(LiftingArm.transform.localPosition.y / 10);
        }
        
    }

    public override void AgentAction(float[] act, string textAction)
    {
        int action = Mathf.FloorToInt(act[0]);

        switch (action)
        {
            case 1:
                SorterArm.GetComponent<SortingArm>().counterClockwise();
                break;

            case 2:
                SorterArm.GetComponent<SortingArm>().clockwise();
                break;

            case 3:
                LiftingArm.GetComponent<LiftingArm>().moveUp();
                break;

            case 4:
                LiftingArm.GetComponent<LiftingArm>().moveDown();
                break;

            default:
                //print("Default action...");
                break;
        }


        if (Ball.GetComponent<Ball>().giveReward)
        {
            SetReward(1);
            Monitor.Log("Reward for " + agentName, GetReward());
            ResetTeacher();
        }
        else if (Ball.GetComponent<Ball>().stopFlag)
        {
            SetReward(-0.1f);
            Monitor.Log("Reward for " + agentName, GetReward());
            ResetTeacher();
        }


    }

    public override void AgentReset()
    {
       
    }

    public override void AgentOnDone()
    {
        
        
    }

    public void ResetTeacher()
    {
        Ball.GetComponent<Ball>().reset();
        SorterArm.GetComponent<SortingArm>().reset();
        LiftingArm.GetComponent<LiftingArm>().reset();
    }



}

