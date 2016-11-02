
import java.io.*;
import java.util.Arrays;
import java.util.Random;
import java.util.Scanner;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.trees.J48;
import weka.core.*;
import weka.clusterers.*;
import weka.filters.*;
import weka.filters.unsupervised.attribute.Remove;
import weka.core.Instances;
import static weka.core.Instances.test;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Discretize;
import weka.filters.unsupervised.attribute.NumericToNominal;

/**
 * This class shows how to perform a "classes-to-clusters"
 * evaluation like in the Explorer using EM. The class needs as
 * first parameter an ARFF file to work on. The last attribute is
 * interpreted as the class attribute. 
 * This code is based on the method "startClusterer" of the 
 * "weka.gui.explorer.ClustererPanel" class and the 
 * "evaluateClusterer" method of the "weka.clusterers.ClusterEvaluation" 
 * class.
 *
 * @author  Ikhwanul Muslimin
 */
public class Tucil2AI {
  
    Instances dataset;
    Classifier cls;

    public Tucil2AI() {

    }
    //Read data set
    public void readDataSet(String filename) throws IOException, Exception{
        BufferedReader inputReader = null;

        try {
          inputReader = new BufferedReader(new FileReader(filename));
          dataset = new Instances(inputReader);
          dataset.setClassIndex(dataset.numAttributes() - 1);
        } catch (FileNotFoundException ex) {
          System.err.println("File not found: " + filename);
        }

    }

    //numeric to nominal
    public void useFilterNtoN() throws IOException, Exception {
    //load training instances

        NumericToNominal convert= new NumericToNominal();
        String[] options= new String[2];
        options[0]="-R";
        options[1]="1-2";  //range of variables to make numeric

        convert.setOptions(options);
        convert.setInputFormat(dataset);
        System.out.println("BEFORE FILTER NtoN");
        System.out.println(dataset);
        Instances newData=Filter.useFilter(dataset, convert);
        System.out.println("AFTER FILTER NtoN");
        System.out.println(newData);
    }

    public void useFilterDiscretize() throws IOException, Exception {
    //load training instances

        Discretize convert= new Discretize();
        String[] options= new String[2];
        options[0]="-R";
        options[1]="1-2";  //range of variables to make numeric

        convert.setOptions(options);
        convert.setInputFormat(dataset);
        System.out.println("BEFORE FILTER Descretize");
        System.out.println(dataset);
        Instances newData=Filter.useFilter(dataset, convert);
        System.out.println("AFTER FILTER Descretize");
        System.out.println(newData);
    }  

    //Cross Fold Validation
    public void train10Fold(Classifier clsf, int folds) throws Exception {
        Evaluation eval = new Evaluation(dataset);
        Random rand = new Random(1);  // using seed = 1
        eval.crossValidateModel(clsf, dataset, folds, rand);
        // output evaluation
        System.out.println(eval.toSummaryString());
        System.out.println(eval.toClassDetailsString());
        System.out.println(eval.toMatrixString());
    }
    
    public void trainFull(Classifier clsf) throws Exception {
        // output evaluation
        Evaluation eval = new Evaluation(dataset);
        eval.evaluateModel(clsf, new Instances(dataset));
        System.out.println(eval.toSummaryString());
        System.out.println(eval.toClassDetailsString());
        System.out.println(eval.toMatrixString());
    }
    
    public void saveModel(String filename) throws Exception {
	SerializationHelper.write(filename, cls);
        System.out.println("Model saved.");
    }
    
    public void loadModel(String filename) throws Exception {
        cls = (Classifier) SerializationHelper.read(filename);
        System.out.println("Model loaded.");
    }
    
    public void createInstance() {
        Scanner sc = new Scanner(System.in);
        Instances test = new Instances(dataset,0);

        double[] values = new double[test.numAttributes()];
        values[0] = sc.nextFloat();
        values[1] = sc.nextFloat();
        values[2] = sc.nextFloat();
        values[3] = sc.nextFloat();
        Instance inst = new DenseInstance(1.0, values);
        test.add(inst);

       test.setClassIndex(test.numAttributes() - 1);
    }
    public static void main(String[] args) {
        int pil;
        String filename = "iris.arff";
        Tucil2AI test = new Tucil2AI();
        //READ DATASET
        try {
            test.readDataSet(filename);
        } catch(Exception e) {

        }
        System.out.println("Dataset loaded.");
        do {
            //MENU
            System.out.println("\nSilakan pilih satu:");
            System.out.println("1. Filter Discretize");
            System.out.println("2. Filter Numeric to Nominal");
            System.out.println("3. 10 Cross Folds Validation");
            System.out.println("4. Full training Validation");
            System.out.println("5. Instance baru");
            System.out.println("6. Save model");
            System.out.println("7. Load model");
            System.out.println("8. Keluar");
            Scanner in = new Scanner(System.in);
            pil = in.nextInt();
            if (pil==1 ) {
                //FILTER Discretize
                try {
                    test.useFilterDiscretize();
                } catch(Exception e) {
                    System.out.println("Gagal melakukan filter Discretize");
                }
            } else
            if (pil==2) {
                  //FILTER Numeric to Nominal
                try {
                    test.useFilterNtoN();
                } catch(Exception e) {
                    System.out.println("Gagal melakukan filter Numeric to Nominal");
                }
            } else
            if (pil==3) {
                System.out.println("10 Cross Validastion dengan Classifier J48");
                //10 Cross Val Split
                try {
                    test.cls = new J48();
                    test.train10Fold(test.cls, 10);
                } catch(Exception e) {
                    System.out.println("Operasi gagal");
                    System.out.println(e);
                }
            } else
            if (pil==4){
                 //Full train
                try {
                    System.out.println("Full training dengan Classifier J48");
                    test.cls = new J48();
                    test.cls.buildClassifier(test.dataset);
                    test.trainFull(test.cls);
                } catch(Exception e) {
                    System.out.println("Operasi gagal");
                    System.out.println(e.toString());
                }
            } else
            if (pil==5) {
                test.createInstance();
            } else
            if (pil==6) {
                try {
                    test.saveModel("savedmodel.model");
                } catch (Exception e) {
                    System.out.println("Gagal menyimpan.");
                    System.out.println(e);
                }
            } else
            if (pil==7) {
                try {
                    test.loadModel("savedmodel.model");
                } catch (Exception e) {
                    System.out.println("Gagal memuat.");
                    System.out.println(e);
                }
            }
        } while (pil!=8);
    }
}
