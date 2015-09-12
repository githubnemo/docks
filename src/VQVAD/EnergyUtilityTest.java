package VQVAD;

import static org.junit.Assert.*;

import org.junit.Test;

import edu.cmu.sphinx.frontend.DoubleData;

public class EnergyUtilityTest {

	@Test
	public void testStd() {
		double res;

		// std([1,2,3,4]) = 1.2910
		res = EnergyUtility.std(new DoubleData(new double[]{1,2,3,4}));
		assertEquals(1.2910, res, 0.0001);

		// std([0,0,0,0]) = 0
		res = EnergyUtility.std(new DoubleData(new double[]{0,0,0,0}));
		assertEquals(0, res, 0.0001);
	}

	@Test
	public void testComputeEnergy() {
		double res;

		// 20 * log10(std([1,2,3,4]) + eps) = 2.2185
		res = EnergyUtility.computeEnergy(new DoubleData(new double[]{1,2,3,4}));
		assertEquals(2.2185, res, 0.0001);

		// 20 * log10(std([1,1,1,1]) + eps) = -313.07
		res = EnergyUtility.computeEnergy(new DoubleData(new double[]{1,1,1,1}));
		assertEquals(-313.07, res, 0.01);

		// 20 * log10(std([0,0,0,0]) + eps) = -313.07
		res = EnergyUtility.computeEnergy(new DoubleData(new double[]{0,0,0,0}));
		assertEquals(-313.07, res, 0.01);
	}

}