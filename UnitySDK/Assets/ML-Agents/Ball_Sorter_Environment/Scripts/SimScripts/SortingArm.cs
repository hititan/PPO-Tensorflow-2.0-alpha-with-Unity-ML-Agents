using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SortingArm : MonoBehaviour
{

    public float speed = 360; // bei 0.01 s fixed time step = 1 s for 360 degrees
    public Vector3 rotation;

    // Use this for initialization
    void Start()
    {
        rotation = this.transform.eulerAngles;
    }

    // Update is called once per frame
    void FixedUpdate()
    {
        // if (Input.GetKey(KeyCode.LeftArrow))
        // {
        //     counterClockwise();
        //     print (Time.fixedDeltaTime);
        //     print (Time.deltaTime);
        //     print(Application.targetFrameRate);
        // }
        //print(this.transform.localRotation.eulerAngles.y);
    }

    public void counterClockwise()
    {
        this.transform.Rotate(Vector3.down * Time.fixedDeltaTime * speed);
        
    }

    public void clockwise()
    {
        this.transform.Rotate(Vector3.up * Time.fixedDeltaTime * speed);
        
    }

    public void reset()
    {
        this.transform.rotation = new Quaternion();
        //this.transform.Rotate(Vector3.up, Random.Range(-90, 90));
        
    }
}
