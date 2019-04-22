
import org.datavec.api.records.reader.impl.csv.CSVRecordReader
import org.datavec.api.split.FileSplit
import org.datavec.api.util.ClassPathResource
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

import scala.io.Source

object DataReader {

  def readFromFiles(labelfile: String, input: String): (INDArray, INDArray) = {
    var labels = Array[Double]()
    var images = Array[String]()
    var total = 0
    for (line <- Source.fromFile(labelfile).getLines) {
      var m = line.split(",")
      images = images :+ m(0)
      labels = labels :+ m(1).toDouble
      total += 1
    }
    var y = Nd4j.create(labels, Array[Int](total, 1))//(labels, new int[]{total,1})

    val recordReaderinput = new CSVRecordReader(0, ",")
    val pathsourceinput = new ClassPathResource(input).getFile
    val file1 = new FileSplit(pathsourceinput)
    recordReaderinput.initialize(file1)
    var concVal = recordReaderinput.next().toArray().map(_.toString).map(_.toFloat)
    var x = Nd4j.create(Array(concVal))
    while (recordReaderinput.hasNext()) {
      concVal = recordReaderinput.next().toArray().map(_.toString).map(_.toFloat)
      x = Nd4j.concat(0,x,Nd4j.create(Array(concVal)))
    }
    x = Nd4j.hstack(x, Nd4j.ones(total,1))

    (x, y)
  }

  def readFromAFile(infile: String, delimiter: String): (INDArray, INDArray) = {

    val numLinesToSkip = 0
    /**
      *  Reading input and labels from the corresponding data files
      */
    val recordReaderinput = new CSVRecordReader(numLinesToSkip, delimiter)
    val pathsourceinput = new ClassPathResource(infile).getFile
    val file1 = new FileSplit(pathsourceinput)
    recordReaderinput.initialize(file1)

    var concVal = recordReaderinput.next().toArray().map(_.toString).map(_.toFloat)
    var x = Nd4j.create(Array(concVal.slice(1,concVal.length)))
    var y = Nd4j.create(Array(concVal.slice(0,1)))

    /**
      * Storing Data files in  INDArray using n4dj, for running logistic regression
      */
    var m = 4999
    var i = 0
    while (recordReaderinput.hasNext() && i < m) {
      concVal = recordReaderinput.next().toArray().map(_.toString).map(_.toFloat)
      x = Nd4j.concat(0,x,Nd4j.create(Array(concVal.slice(1,concVal.length))))
      y = Nd4j.concat(0,y,Nd4j.create(Array(concVal.slice(0,1))))
      i += 1
    }
    (x,y)
  }

}

object main extends App {

  /// READ LABELS AND FILES

  import DataReader._
  ///*
  var (x,y) = readFromAFile("mnist_test.csv", ",")
  var rows = Nd4j.shape(x).toSeq(0)
  //(x,y) = (x.get(NDArrayIndex.interval(0,rows/2)), y.get(NDArrayIndex.interval(0,rows/2)))
  val iterations = 2000
  var lr = 0.00000005f
  val batchsize = 1667
  var noc = 10
  //*/
  //print(Nd4j.shape(x).toSeq, x.getRow(0), y.sumNumber().doubleValue())

  /*
  var (x,y) = readFromAFile("satimage.scale.tr.csv", " ")
  var rows = Nd4j.shape(x).toSeq(0)
  val iterations = 10000
  var lr = 0.005f
  val batchsize = 1000
  var noc = 6
  */
  /*
  var (x,y) = readFromAFile("vehicle.scale.txt.csv", " ")
  val iterations = 2000
  var lr = 0.0005f
  val batchsize = 450
  val d = 4
  var noc = 6
  */

  var ndims = x.columns()
  print(x.shape().toSeq)

  ///*
  val model = new MultiSVM()
  model.fit(y, x, lr, iterations, batchsize, noc)
  val y_new = model.predict(x)
  val acc = model.accuracy(y_new, y)
  println(acc * 100)//*/

}
