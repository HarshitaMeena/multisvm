import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.NDArrayIndex
//import org.nd4j.linalg.indexing.BooleanIndexing

class MultiSVM() {

  var svm_weights: INDArray = _
  def shuffleData(x: INDArray, y: INDArray): INDArray = {
    val stackedData = Nd4j.hstack(x,y)
    Nd4j.shuffle(stackedData, 1)
    //print(combineddata.getRow(0))
    stackedData
  }


  def backward(X: INDArray, y: INDArray): INDArray = {
      var N = X.shape().toSeq(0)
      var k = svm_weights.shape().toSeq(0)
      var ones = Nd4j.ones(k,N)
      var delta = Nd4j.zeros(k,N)
      for (i <- 0 to N-1) {
        var j = y.getInt(i,0)
        delta.putScalar(Array[Int](j, i), 1)
      }

      var WXT = svm_weights.mmul(X.transpose())
      //println(WXT)
      var intermediate = ones.sub(delta).add(WXT)
      var maxArgVal = Nd4j.zeros(N)
      var c = intermediate.columns()
      var r = intermediate.rows()

      for (j <- 0 to c-1) {
        var maxi = intermediate.getDouble(0,j)
        var index = 0
        for (i <- 1 to r-1) {
          if (maxi < intermediate.getDouble(i,j)) {
            maxi = intermediate.getDouble(i,j)
            index = i
          }
        }
        //print(index)
        maxArgVal.putScalar(Array[Int](j), index)
      }


      //var maxArgVal = Nd4j.getExecutioner().exec(new IAMax(ones.sub(delta).add(WXT)),0)
      //println(maxArgVal)

      var grad = Nd4j.zeros(k, X.shape().toSeq(1))
      for (i <- 0 to k-1) {
          grad.putRow(i, svm_weights.getRow(i))
          if (i == 0) {
            //println(grad.getDouble(0,784))
          }
          var pos = 0
          var neg = 0
          //println(grad.getRow(i))
          for (j <- 0 to N-1) {

              if (maxArgVal.getInt(j) == i) {
                  //print("(", maxArgVal.getInt(j), i, y.getInt(j,0), ")")
                  var intermediate = grad.getRow(i).add(X.getRow(j))
                  grad.putRow(i, intermediate)
                  pos += 1
              }
              if (y.getInt(j,0) == i) {
                  //print("(", maxArgVal.getInt(j), i, y.getInt(j,0), ")")
                  var intermediate = grad.getRow(i).sub(X.getRow(j))
                  grad.putRow(i, intermediate)
                  neg -= 1
              }

              /*if (y.getInt(j,0) == i && maxArgVal.getInt(j) == i) {
                 println("Yayy")
              }*/
          }
          if (i == 0) {
            //println(pos, neg)
          }
      }
      //println(grad.shape().toSeq)
      //println(grad.getDouble(0,784))
      //println(grad.getRow(1))
      grad
  }


  def predict(input: INDArray): INDArray = {

      var X_intercept = Nd4j.hstack(input, Nd4j.ones(input.rows(), 1))
      var prob = svm_weights.mmul(X_intercept.transpose())
      //print(prob)
      //prob = prob.reshape(prob.columns(), prob.rows())
      var tags = Nd4j.zeros(input.rows(), 1)
      var c = prob.columns()
      var r = prob.rows()

      for (j <- 0 to c-1) {
          var maxi = prob.getDouble(0,j)
          var index = 0
          for (i <- 1 to r-1) {
              if (maxi < prob.getDouble(i,j)) {
                 maxi = prob.getDouble(i,j)
                  index = i
              }
          }
          //print(index)
          tags.putScalar(Array[Int](j,0), index)
      }
      //var tags = Nd4j.getExecutioner().exec(new IAMax(prob), 0)
      //print(tags)
      tags
  }


  def fit(y_true: INDArray, input: INDArray, learn_rate: Double, max_iters: Int, batchsize: Int, noofclasses: Int): Unit = {

    var t1 = System.nanoTime
    var K = noofclasses
    var X_intercept  = Nd4j.hstack(input, Nd4j.ones(input.rows(), 1))
    var (n, d) = (X_intercept.shape().toSeq(0), X_intercept.shape().toSeq(1))

    svm_weights = Nd4j.zeros(K, d)

    var j = 1
    val totalbatches = math.ceil(X_intercept.rows / batchsize.toFloat).toInt
    var combinedData = shuffleData(X_intercept, y_true)
    for (i <- 0 to max_iters) {
      var t2 = System.nanoTime
      if (j == totalbatches+1) {
        combinedData = shuffleData(X_intercept, y_true)
        j = 1
      }
      val subpart = combinedData.get(NDArrayIndex.interval((j - 1) * batchsize, math.min(j * batchsize, combinedData.rows)))
      val xbatch = subpart.get(NDArrayIndex.all(), NDArrayIndex.interval(0, subpart.columns - 1))
      val ybatch = subpart.get(NDArrayIndex.all(), NDArrayIndex.point(subpart.columns - 1))

      var gradient = backward(xbatch, ybatch)
      //print(gradient)
      svm_weights = svm_weights.add(gradient.mul(-learn_rate))
      //
      //print(svm_weights.getColumn(784))
      j += 1
      //print(i, (System.nanoTime-t2)/1e9, xbatch.shape().toSeq)
    }
    val duration = (System.nanoTime - t1) / 1e9
    println()
    println("TOTAL TIME FOR PROCESSING for batchsize: ",  batchsize, " and iterations ", max_iters , " with cost ",
      computeCost(X_intercept,y_true, 1), " is ", duration)

  }


  def accuracy(y: INDArray, y_true: INDArray): Double = {
    var count = 0.0
    //print(y)
    //print(y.shape().toSeq, y_true.shape().toSeq)
    for (i <- 0 to y.size(0)-1) {
      if (y.getDouble(i,0) == y_true.getDouble(i,0)) {
        count += 1
      }
    }
    //print(y, y_true, count)
    count/y.rows()
  }


  def computeCost(X: INDArray, y: INDArray, C: Double): Double  = {
    var magnitude = 0.5*Nd4j.diag(svm_weights.mmul(svm_weights.transpose())).sumNumber().doubleValue()

    var N = X.shape().toSeq(0)
    var k = svm_weights.shape().toSeq(0)
    var ones = Nd4j.ones(k,N)
    var delta = Nd4j.zeros(k,N)
    for (i <- 0 to N-1) {
      var j = y.getInt(i,0)
      delta.putScalar(Array[Int](j, i), 1)
    }

    var WXT = svm_weights.mmul(X.transpose())
    var intermediate = ones.sub(delta).add(WXT)
    var maxArgVal = Nd4j.zeros(N)
    var c = intermediate.columns()
    var r = intermediate.rows()

    for (j <- 0 to c-1) {
      var maxi = intermediate.getDouble(0,j)
      var index = 0
      for (i <- 1 to r-1) {
        if (maxi < intermediate.getDouble(i,j)) {
          maxi = intermediate.getDouble(i,j)
          index = i
        }
      }
      //print(index)
      maxArgVal.putScalar(Array[Int](j), maxi)
    }

    var losspart2 = maxArgVal.sumNumber().doubleValue()

    var wd = WXT.mul(delta)
    var losspart3 = wd.sumNumber().doubleValue()
    //println(magnitude, losspart2, losspart3, magnitude + C*(losspart2-losspart3), C)
    magnitude + C*(losspart2-losspart3)
  }

}
