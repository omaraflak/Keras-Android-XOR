package aflak.me.tensorflowlitexor;

import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;

import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import java.util.Arrays;

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
