package com.example.myai;

import static org.deeplearning4j.nn.api.OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT;
import static org.nd4j.linalg.activations.Activation.SIGMOID;
import static org.nd4j.linalg.activations.Activation.SOFTMAX;
import static org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD;
import androidx.appcompat.app.AppCompatActivity;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import android.os.Bundle;
import java.io.File;
import java.io.IOException;

public class MainActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        int height = 1080;    // высота изображения
        int width = 1920;     // ширина изображения
        int channels = 3;     // каналы изображения (3 для RGB, 1 для Grayscale)
        int outputNum = 2;
        INDArray labelsList = null;
        INDArray featureList = null;


        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(123)
                .optimizationAlgo(STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Adam())
                .list()
                .layer(0, new ConvolutionLayer.Builder(5, 5)
                        .nIn(channels)
                        .stride(1, 1)
                        .nOut(width)
                        .activation(SIGMOID)
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .layer(1, new DenseLayer.Builder().activation(SIGMOID).nOut(height).build())
                .layer(2, new OutputLayer.Builder(NEGATIVELOGLIKELIHOOD)
                        .nOut(outputNum)
                        .activation(SOFTMAX)
                        .build())
                .build();

        // Здесь вы можете инициализировать свою модель с помощью конфигурации
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

        //2. Загрузка DataFiles
        File folder = new File(getFilesDir(), "dataset");
        File[] listOfFiles = folder.listFiles();
        for (File file : listOfFiles) {
            if (file.isFile()) {
                // Загрузка изображения
                NativeImageLoader loader = new NativeImageLoader(height, width, channels);
                INDArray image = null;
                try {
                    image = loader.asMatrix(file);
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }

                // Нормализация изображения
                DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
                scaler.transform(image);

                // Добавление изображения в список признаков

                featureList.add(image);

                // Парсинг имени файла для получения меток
                String fileName = file.getName();
                fileName = fileName.replace("files:", ""); // Удаление "files:"
                String[] splitName = fileName.split("_");
                double x = Double.parseDouble(splitName[0]);
                double y = Double.parseDouble(splitName[1].split("\\.")[0]);

                // Добавление меток в список меток

                labelsList.add(Nd4j.create(new double[]{x, y}));
            }
        }

        //3. Обучение ИИ

// Разделение данных на обучающую и тестовую выборки
        int batchSize = 128; // Размер пакета для обучения
        int numEpochs = 10; // Количество эпох для обучения

        DataSet allData = new DataSet(featureList, labelsList);
        allData.shuffle();
        SplitTestAndTrain testAndTrain = allData.splitTestAndTrain(0.8);  // 80% данных для обучения, 20% для тестирования

        DataSet trainingData = testAndTrain.getTrain();
        DataSet testData = testAndTrain.getTest();

// Обучение модели
        for( int i=0; i<numEpochs; i++ ){
            model.fit(trainingData);
        }

// Оценка модели
        Evaluation eval = new Evaluation(outputNum); // Создание объекта для оценки модели
        INDArray output = model.output(testData.getFeatures()); // Получение выходных данных модели для тестовой выборки
        eval.eval(testData.getLabels(), output); // Оценка модели
        int screenWidth = 1920;
        int screenHeight = 1080;
        INDArray xCoords = output.getColumn(0).mul(screenWidth); // Первый столбец - это предсказания по оси x
        INDArray yCoords = output.getColumn(1).mul(screenHeight); // Второй столбец - это предсказания по оси y

// Вывод результатов оценки
        System.out.println(eval.stats());

        //4. Сохранение обученной модели

        String modelFilename = "model.zip";

// Сохранение модели
        File locationToSave = new File(modelFilename);
        boolean saveUpdater = true; // Сохранение информации для дальнейшего обучения
        try {
            ModelSerializer.writeModel(model, locationToSave, saveUpdater);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }


// 5. Загрузка обученной модели

        try {
            model = ModelSerializer.restoreMultiLayerNetwork(modelFilename, saveUpdater);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }


    }
}