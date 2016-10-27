import java.io.File

import org.apache.commons.math3.distribution.MultivariateNormalDistribution
import org.apache.commons.math3.linear._
import org.apache.commons.math3.random.MersenneTwister

import scala.io.Source


val A = new Array2DRowRealMatrix(3, 6)
val r = new scala.util.Random(0)

for(i <- 0 until A.getRowDimension())
  for(j <- 0 until A.getColumnDimension())
    A.setEntry(i, j, r.nextDouble())

A.setEntry(0, 0,
  A.getEntry(A.getRowDimension() - 1, A.getColumnDimension() - 1))

val B = A.getSubMatrix(0, 2, 0, 2)

A.setSubMatrix(B.getData(), 0, 1)

val C = A.transpose().multiply(B)

val solver = new LUDecomposition(B).getSolver()
val a = A.getColumnVector(0)
val x = solver.solve(a)

val diag1 = Array(3.0,4,5)
val diag = new DiagonalMatrix(diag1)


val src = Source.fromFile("/Users/yfang/workspace/asi/data/factor_correlation")
val corre = src.getLines().map(_.split(",")).map(_.map(_.toDouble)).toArray
corre.toArray.deep

val src1 = Source.fromFile("/Users/yfang/workspace/asi/data/annual_std")
val std = src1.getLines().toArray.map(x=>x.toDouble/100)
std.deep

val std_diag = new DiagonalMatrix(std).scalarMultiply(1/Math.sqrt(12))
val corre_matrix = new Array2DRowRealMatrix(corre)
val cov = std_diag.multiply(corre_matrix).multiply(std_diag)

def printToFile(f: java.io.File)(op: java.io.PrintWriter => Unit) {
  val p = new java.io.PrintWriter(f)
  try { op(p) } finally { p.close() }
}

val data = Array("Five","strings","in","a","file!")
printToFile(new File("/tmp/example.txt")) { p =>
  data.foreach(p.println)
}


printToFile(new File("/Users/yfang/workspace/asi/data/covariance")) { p =>
  cov.getData.map(_.mkString(",")).foreach(p.println)
}

val annual_return = Source.fromFile("/Users/yfang/workspace/asi/data/annual_return").getLines().toArray.map(x=>x.toDouble/1200)
annual_return.deep

val rand = new MersenneTwister(1341L)
val multivariateNormal = new MultivariateNormalDistribution(rand, annual_return,
  cov.getData)

for(i<- 1 until 10000){
  println(multivariateNormal.sample().deep)
}
multivariateNormal.sample().deep

