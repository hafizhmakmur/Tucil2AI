import java.util.Scanner;
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.util.Random;
import java.util.ArrayList;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.supervised.attribute.Discretize;
import weka.core.Utils;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.classifiers.trees.J48;
import weka.classifiers.Evaluation;
import weka.classifiers.Classifier;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils.DataSink;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.DenseInstance;

class MachineLearner {

	Instances dataset;

	Instances test;

	Classifier cls;

	public MachineLearner(String path) {
		try {
			readDataSet(path);
			test = new Instances(dataset,0);
			test.setClassIndex(test.numAttributes() - 1);
		} catch (Exception e) {
			System.out.println("What the heck : " + e);
		}
	}

	public MachineLearner() {
		try {
			readDataSet();
			test = new Instances(dataset,0);
			test.setClassIndex(test.numAttributes() - 1);
		} catch (Exception e) {
			System.out.println("What the heck : " + e);
		}
	}

	public void readDataSet() throws Exception {
		dataset = DataSource.read("D:\\soft\\Weka-3-8\\data\\iris.arff");
		dataset.setClassIndex(dataset.numAttributes() - 1);
	}

	public void readDataSet(String path) throws Exception {
		dataset = DataSource.read(path);
		dataset.setClassIndex(dataset.numAttributes() - 1);
	}

	public void useFilterDiscretize(String option) throws Exception {

		Discretize filter = new Discretize();

		String[] options = Utils.splitOptions(option);
		filter.setOptions(options);
		
		filter.setInputFormat(dataset);

		dataset = Filter.useFilter(dataset,filter);
	}

	public void useFilterNtoN(String option) throws Exception {

		NumericToNominal filter = new NumericToNominal();

		String[] options = Utils.splitOptions(option);
		filter.setOptions(options);
		
		filter.setInputFormat(dataset);

		dataset = Filter.useFilter(dataset,filter);
	}

	public void train10Fold(String option) throws Exception {
		Evaluation eval = new Evaluation(dataset);

		cls = new J48();
		cls.buildClassifier(dataset);
/*		String[] options = Utils.splitOptions(option);
		tree.setOptions(options);
*/		
		eval.crossValidateModel(cls, dataset, 10, new Random(1));
//		System.out.println(eval.toSummaryString("\nResults\n\n", false));
	}

	public void trainFull(String option) throws Exception {

		test = new Instances(dataset);
		
		// train classifier
		cls = new J48();
		cls.buildClassifier(dataset);
/*		String[] options = Utils.splitOptions(option);
		cls.setOptions(options);
*/		// evaluate classifier and print some statistics
		Evaluation eval = new Evaluation(dataset);
		eval.evaluateModel(cls, test);
//		System.out.println(eval.toSummaryString("\nResults\n\n", false));
	
	}

	public void saveModel(String filename) throws Exception {
		SerializationHelper.write(filename, cls);
	}

	public void loadModel(String filename) throws Exception {
		cls = (Classifier) SerializationHelper.read(filename);
	}

	public void createInstance() {
		Scanner sc = new Scanner(System.in);
/*
		Attribute sepallength = new Attribute("sepallength");
		Attribute sepalwidth = new Attribute("sepalwidth");
		Attribute petallength = new Attribute("petallength");
		Attribute petalwidth = new Attribute("petalwidth");
		ArrayList<String> labels = new ArrayList<String>();
		labels.add("Iris-setosa");
		labels.add("Iris-versicolor");
		labels.add("Iris-virginica");
		Attribute cls = new Attribute("class", labels);
		ArrayList<Attribute> attributes = new ArrayList<Attribute>();
		attributes.add(sepallength);
		attributes.add(sepalwidth);
		attributes.add(petallength);
		attributes.add(petalwidth);
		attributes.add(cls);
		test = new Instances("Test-dataset", attributes, 0);
*/
		test = new Instances(dataset,0);
//		System.out.println(test);
//		System.out.println(dataset);

		double[] values = new double[test.numAttributes()];
		values[0] = sc.nextFloat();
		values[1] = sc.nextFloat();
		values[2] = sc.nextFloat();
		values[3] = sc.nextFloat();
//		values[4] = 1.0;
//		values[4] = test.attribute(4).addStringValue("Iris-setosa");
//		System.out.println(test.attribute(4).addStringValue("Iris-setosa"));
		Instance inst = new DenseInstance(1.0, values);
		test.add(inst);


//		test.insertAttributeAt(dataset.attribute(dataset.numAttributes() - 1), test.numAttributes());
		test.setClassIndex(test.numAttributes() - 1);
	}

	public void classify() throws Exception {
		// create copy
		Instances labeled = new Instances(test);
		labeled.setClassIndex(labeled.numAttributes()-1);

		// label instances
		for (int i = 0; i < test.numInstances(); i++) {
		//	System.out.println(test.numInstances());
		//	System.out.println(test.instance(i).toString(test.classIndex()));
			double clsLabel = cls.classifyInstance(test.instance(i));
		//	clsLabel = 2.3;
			labeled.instance(i).setClassValue(clsLabel);
		//	System.out.println(clsLabel);
			System.out.println(test.classAttribute().value((int) clsLabel));
		}

//		System.out.println(labeled);
//		DataSink.write("labeled.arff", labeled);
	}

	public static void main(String[] args) {
		MachineLearner weka = new MachineLearner();
		
		try {
			weka.readDataSet();	
			weka.useFilterDiscretize("-R first-last -precision 6");			
		//	weka.useFilterNtoN("-R first-last");
			weka.train10Fold("-C 0.25 -M 2");
		//	weka.trainFull("-C 0.25 -M 2");
		//	weka.saveModel("simpen.model");
		//	weka.loadModel("simpen.model");
			weka.createInstance();
			weka.classify();
		} catch (Exception e) {
			System.out.println("The hell indeed : " + e);
			e.printStackTrace();
		}
	}
}