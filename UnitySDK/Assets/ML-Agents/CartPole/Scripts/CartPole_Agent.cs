using System.Collections;
using System.Collections.Generic;
//using System;
using UnityEngine;
using MLAgents;

public class CartPole_Agent: Agent {

    public GameObject Cart;
    public GameObject Pole;
    Rigidbody rbCart;
    Rigidbody rbPole;

    public bool useDone = true;
    public float angelTreshold = 12;

    public override void InitializeAgent()
    {
        rbCart = Cart.GetComponent<Rigidbody>();
        rbPole = Pole.GetComponent<Rigidbody>();
    }
    public override void CollectObservations()
    {
        AddVectorObs(Cart.transform.localPosition.z);
        AddVectorObs(rbCart.velocity.z);
        AddVectorObs(getAngel() * 3.14f / 180f);
        AddVectorObs(rbPole.angularVelocity.x * 1 / 10f);
    }

    public override void AgentAction(float[] vectorAction, string textAction)
	{
        int act = Mathf.FloorToInt(vectorAction[0]);

        if (this.brain.name == "PlayerBrain")
        {
            switch (act)
            {
                case 2:
                    rbCart.AddForce(new Vector3(0, 0, 10), ForceMode.Force);
                    break;
                case 1:
                    rbCart.AddForce(new Vector3(0, 0, -10), ForceMode.Force);
                    break;
                case 0:
                    break;
                default:
                    break;
            }
        }
        else
        {
            switch (act)
            {
                case 0:
                    rbCart.AddForce(new Vector3(0, 0, 10), ForceMode.Force);
                    break;
                case 1:
                    rbCart.AddForce(new Vector3(0, 0, -10), ForceMode.Force);
                    break;
                default:
                    break;
            }

        }
        
        if(useDone)
        {
            if (getAngel() < -angelTreshold || getAngel() > angelTreshold)
            {
                Done();
            }

            if (Cart.transform.localPosition.z < -2.4f || Cart.transform.localPosition.z > 2.4f)
            {
                Done();
            }
        }
     
        AddReward(1.0f);
    }

    public override void AgentReset()
    {
        rbCart.velocity = new Vector3(0, 0, 0);
        rbPole.angularVelocity = new Vector3(0, 0, 0);
        
        rbCart.transform.localPosition = new Vector3(0, 0, Random.Range(-0.1f, 0.1f));
        rbPole.rotation = Quaternion.identity;
        rbPole.transform.localRotation= Quaternion.Euler(Random.Range(-12f, 12f), 0, 0);
    }

    private float getAngel()
    {
        float angel;
        angel = Pole.transform.rotation.eulerAngles.x;

        if (angel > 180)
        {
            angel = angel - 360;
        }
        return angel;
    }

    public override void AgentOnDone()
    {

    }
}
