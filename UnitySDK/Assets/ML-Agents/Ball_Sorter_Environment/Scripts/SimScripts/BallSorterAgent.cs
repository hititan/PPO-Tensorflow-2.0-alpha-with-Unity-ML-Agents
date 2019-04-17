using System;
using System.Collections;
using UnityEngine;
using MLAgents;


public class BallSorterAgent : Agent
{
    public GameObject Ball;
    public GameObject SorterArm;
    public GameObject LiftingArm;

    // public bool realWorldMode = false;
    // public float[] controllArray;
    public bool useVisual = false;

    private string agentName;

    public override void InitializeAgent()
    {
        // Set gravity for cm/sec**2  
        Physics.gravity = new Vector3(0, -981.0f, 0);
        agentName = this.gameObject.name;
        // controllArray = new float[5]; 
        // GRPCService.start();
    }

    public override void CollectObservations()
    {
        if (this.useVisual == false)
        {
            foreach (int co in System.Enum.GetValues(typeof(sorterEnums.ballColors)))
            {
                AddVectorObs((int)Ball.GetComponent<Ball>().GetMyColor() == co ? 1.0f : 0.0f);
            }

            AddVectorObs(SorterArm.transform.rotation.eulerAngles.y / 360);
            AddVectorObs(LiftingArm.transform.localPosition.y / 10);
        }


        //if (this.brain.name == "ObsLearningBrain")
        //{
        //if (!realWorldMode)
        //{
        //    foreach (int co in System.Enum.GetValues(typeof(sorterEnums.ballColors)))
        //    {
        //        AddVectorObs((int)Ball.GetComponent<Ball>().GetMyColor() == co ? 1.0f : 0.0f);
        //    }

        //    AddVectorObs(SorterArm.transform.rotation.eulerAngles.y / 360);
        //    AddVectorObs(LiftingArm.transform.localPosition.y / 10);
        //}
        //else if (realWorldMode)
        //{
        //    var observationsArray = GRPCService.getObservations().Observations;
        //    AddVectorObs(observationsArray[0]);
        //    AddVectorObs(observationsArray[1]);
        //    AddVectorObs(observationsArray[2]);
        //    AddVectorObs(observationsArray[3]);
        //    AddVectorObs(observationsArray[4]);
        //}
        //}
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

        //if (realWorldMode)
        //{
        //    //controllArray[0] = SorterArm.transform.rotation.eulerAngles.y / 360;
        //    //controllArray[1] = LiftingArm.transform.localPosition.y / 10;

        //    //var actResponse = GRPCService.setActions(controllArray);

        //    //if (!actResponse.ActionsPerformed)
        //    //{
        //    //    Done();
        //    //}
        //}
        //else if (!realWorldMode)
        //{
        if (Ball.GetComponent<Ball>().giveReward)
        {
            SetReward(1);
            Monitor.Log("Reward for " + agentName, GetReward());
            Done();
        }
        else if (Ball.GetComponent<Ball>().stopFlag)
        {
            SetReward(-0.1f);
            Monitor.Log("Reward for " + agentName, GetReward());
            Done();
        }
        //SetReward(-0.5f);
        //}
    }

    public override void AgentReset()
    {
        //if(realWorldMode)
        //{
        //    Array.Clear(controllArray, 0, controllArray.Length);
        //    GRPCService.setActions(controllArray);
        //}

        Ball.GetComponent<Ball>().reset();
        SorterArm.GetComponent<SortingArm>().reset();
        LiftingArm.GetComponent<LiftingArm>().reset();
    }

    public override void AgentOnDone()
    {
    }

    private void OnApplicationQuit()
    {
        //GRPCService.stop();
    }

}

