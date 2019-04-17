using System.Collections;
using System.Collections.Generic;
using UnityEngine;


namespace sorterEnums
{
    public enum ballColors
    {
        Red = 1,
        Green = 2,
        Blue = 3
    }
}


public class Ball : MonoBehaviour
{
    private sorterEnums.ballColors color;
    private Color newColor = new Color(0, 0, 0);
    public bool giveReward = false;
    public bool stopFlag = false;

    // Use this for initialization
    void Start()
    {

    }

    // Update is called once per frame
    void FixedUpdate()
    {

    }

    public void reset()
    {
        giveReward = false;
        stopFlag = false;

        this.transform.localPosition = new Vector3(0, 10, 0);
        this.GetComponent<Rigidbody>().velocity = new Vector3(0, 0, 0);

        color = GetRandomEnum<sorterEnums.ballColors>();

        switch (color)
        {
            case sorterEnums.ballColors.Red:
                newColor = new Color(1, 0, 0);
                break;

            case sorterEnums.ballColors.Green:
                newColor = new Color(0, 1, 0);
                break;

            case sorterEnums.ballColors.Blue:
                newColor = new Color(0, 0, 1);
                break;
        }

        this.GetComponent<Renderer>().material.color = newColor;
    }

    void OnCollisionEnter(Collision col)
    {
        var gaObjName = col.gameObject.name;

        if (color.ToString() == gaObjName)
        {
            //print("Right");
            giveReward = true;
        }
        else if (gaObjName != "Slope" && gaObjName != "SideWall 1" && gaObjName != "SideWall 2"
                    && gaObjName != "LiftingArm")
        {
            //print("False");
            //giveReward = false;
            stopFlag = true;
        }
    }

    static T GetRandomEnum<T>()
    {
        System.Array A = System.Enum.GetValues(typeof(T));
        T V = (T)A.GetValue(Random.Range(0, A.Length));
        return V;
    }

    public sorterEnums.ballColors GetMyColor()
    {
        return color;
    }
}
