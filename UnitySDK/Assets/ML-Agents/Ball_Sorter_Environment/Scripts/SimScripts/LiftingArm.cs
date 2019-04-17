using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class LiftingArm : MonoBehaviour
{
    public float Speed = 20;
    public float movement;
    public float upperLimit = 7.0f;
    public float lowerLimit = 4.0f;
    public Vector3 resetPosition;


    // Use this for initialization
    void Start()
    {
        resetPosition = this.transform.localPosition;

        this.transform.localPosition = new Vector3(transform.localPosition.x,
                                                 Mathf.Clamp(this.transform.localPosition.y, lowerLimit, upperLimit),
                                                 transform.localPosition.z);

        movement = Time.fixedDeltaTime * Speed;
    }

    // Update is called once per frame
    void FixedUpdate()
    {}

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

    public void reset()
    {
        this.transform.localPosition = resetPosition;
    }


}
