# How to use a Tensorflow model on Android

This code is a super simple example to understand how to run a Tensorflow (or Keras) model on Android.

## Train the model on a computer

run the python script :

    python index.py
    
When done, the script should have created an `out` folder which contains several files. Among them, **`tensorflow_lite_xor_nn.pb`**, which is the model to export in Android **assets** folder.

## Run the model on Android

**[MainActivity.java](https://github.com/OmarAflak/TensorflowLite-XOR/blob/master/TensorflowLiteXOR/app/src/main/java/aflak/me/tensorflowlitexor/MainActivity.java)**

```java
public class MainActivity extends AppCompatActivity {
    private TensorFlowInferenceInterface inferenceInterface;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // Load model from assets
        inferenceInterface = new TensorFlowInferenceInterface(getAssets(), "tensorflow_lite_xor_nn.pb");

        // run the model for all possible inputs i.e. [0,0], [0,1], [1,0], [1,1]
        for(int i=0 ; i<2 ; i++){
            for(int j=0 ; j<2 ; j++){
                float[] input = {i,j};
                float[] output = predict(input);

                Log.d(getClass().getSimpleName(), Arrays.toString(input)+" -> "+Arrays.toString(output));
            }
        }
    }

    private float[] predict(float[] input){
        // model has only 1 output neuron
        float output[] = new float[1];

        // feed network with input of shape (1,input.length) = (1,2)
        inferenceInterface.feed("dense_1_input", input, 1, input.length);
        inferenceInterface.run(new String[]{"dense_2/Sigmoid"});
        inferenceInterface.fetch("dense_2/Sigmoid", output);

        // return prediction
        return output;
    }
}
```
