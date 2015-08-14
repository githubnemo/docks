package VQVAD;

import java.util.ArrayList;
import java.util.List;

import edu.cmu.sphinx.frontend.BaseDataProcessor;
import edu.cmu.sphinx.frontend.Data;
import edu.cmu.sphinx.frontend.DataProcessingException;
import edu.cmu.sphinx.frontend.DataProcessor;
import edu.cmu.sphinx.frontend.DoubleData;
import edu.cmu.sphinx.frontend.FrontEnd;
import edu.cmu.sphinx.frontend.transform.DiscreteFourierTransform;
import edu.cmu.sphinx.frontend.window.RaisedCosineWindower;

public class SpectralSubtracter {

	protected double sum(double[] v) {
		double sum = 0;
		for (int i=0; i < v.length; i++) sum += v[i];
		return sum;
	}

	protected double mean(double[] v) {
		return sum(v) / v.length;
	}

	protected double round(double v) {
		return Math.floor(v + 0.5);
	}

	protected double min(double[] v) {
		double min = Double.POSITIVE_INFINITY;
		for (int i=0; i < v.length; i++) if (v[i] < min) min = v[i];
		return min;
	}

	protected double max(double[] v) {
		double max = Double.NEGATIVE_INFINITY;
		for (int i=0; i < v.length; i++) if (v[i] > max) max = v[i];
		return max;
	}

	protected double[] slice(double[] v, int start, int end) {
		double[] res = new double[end - start];
		for (int i=start, j=0; i < end; i++, j++) res[j] = v[i];
		return res;
	}

	public double[][] estnoiseg(double[][] powerSpectrum, double frameIncrement) {
		double[][] yf = powerSpectrum;
		double tz = frameIncrement;

		int nr = yf.length;
		int nrf = yf[0].length; // number of frames and freq bins

		double[][] x = new double[nr][nrf]; // initialize output arrays

		double tinc = tz;    //      % second argument is frame increment
		int nrcum=0;           // % no frames so far

		// % default algorithm constants
		double tax=0.0717;      // % noise output smoothing time constant = -tinc/log(0.8) (8)
		double tap=0.152;       // % speech prob smoothing time constant = -tinc/log(0.9) (23)
		double psthr=0.99;      // % threshold for smoothed speech probability [0.99] (24)
		double pnsaf=0.01;      // % noise probability safety value [0.01] (24)
		double pspri=0.5;       // % prior speech probability [0.5] (18)
		double asnr=15;         // % active SNR in dB [15] (18)
		double psini=0.5;       // % initial speech probability [0.5] (23)
		double tavini=0.064;    // % assumed speech absent time at start [64 ms]

		double pslp[] = new double[nrf];
		for (int i=0; i < nrf; i++) pslp[i] = psini; // initialize smoothed speech presence prob

		double[] xt = new double[nrf]; // initialize just in case the first call has no data

		// derived algorithm constants
		double ax=Math.exp(-tinc/tax); // % noise output smoothing factor = 0.8 (8)
		double axc=1-ax;
		double ap=Math.exp(-tinc/tap); // % noise output smoothing factor = 0.9 (23)
		double apc=1-ap;
		double xih1=Math.pow(10, (asnr/10)); // % speech-present SNR
		double xih1r=1/(1+xih1)-1;
		double pfac=(1/pspri-1)*(1+xih1); // % p(noise)/p(speech) (18)


		if (nrcum==0 && nr>0) { //       % initialize values for first frame
			// xt=psini*mean(yf(1:max(1,min(nr,round(1+qq.tavini/tinc))),:),1);  // % initial noise estimate

			// yf(1:max(1,min(nr,round(1+qq.tavini/tinc))),:) => 1:max() rows of yf
			int numRows = (int) Math.max(1, Math.min(nr, round(1+tavini/tinc)));

			// psini*mean(yf(...), 1) => mean over each column of yf(...)
			for(int col=0; col < nrf; col++) {
				for(int row=0; row < numRows; row++) {
					xt[col] += yf[row][col];
				}
				xt[col] /= numRows;
				xt[col] *= psini;
			}
		}

		// % loop for each frame
		// for t=1:nr
		for (int t=0; t < nr; t++) {
			double[] yft = new double[nrf];        // % noisy speech power spectrum
			for (int i=0; i < nrf; i++) yft[i] = yf[t][i]; // yf(t,:);

			// ph1y=(1+pfac*exp(xih1r*yft./xt)).^(-1); % a-posteriori speech presence prob (18)
			double[] ph1y = new double[xt.length];
			for (int i=0; i < xt.length; i++) {
				ph1y[i] = 1 + pfac * Math.exp(xih1r * yft[i] / xt[i]);
			}

			// pslp=ap*pslp+apc*ph1y; % smoothed speech presence prob (23)
			for (int i=0; i < pslp.length; i++) {
				pslp[i] = ap * pslp[i] + apc * ph1y[i];
			}

			// ph1y=min(ph1y,1-pnsaf*(pslp>psthr)); % limit ph1y (24)
			for (int i=0; i < ph1y.length; i++) {
				ph1y[i] = Math.min(ph1y[i], 1-pnsaf*((pslp[i] > psthr) ? 1 : 0));
			}

			// xtr=(1-ph1y).*yft+ph1y.*xt; % estimated raw noise spectrum (22)
			double[] xtr = new double[ph1y.length];
			for (int i=0; i < xtr.length; i++) {
				xtr[i] = (1-ph1y[i]) * yft[i] + ph1y[i] * xt[i];
			}

			// xt=ax*xt+axc*xtr;  % smooth the noise estimate (8)
			for (int i=0; i < xtr.length; i++) {
				xt[i] = ax * xt[i] + axc * xtr[i];
			}

			// x(t,:)=xt;  % save the noise estimate
			x[t] = xt;
		}
		return x;
	}

	// function [ss,gg,tt,ff,zo]=specsub(si,fsz,pp)
	public double[] specsub(double[] signal, float sampleRate) {

		float fs = sampleRate; // % sample frequency
		final double[] s = signal; // s=si(:);
	// % default algorithm constants


	double of=2;   // % overlap factor = (fft length)/(frame increment)
	double ti=16e-3; //  % desired frame increment (16 ms)
	double ri=0;      // % round ni to the nearest power of 2
	int qq_g=1;       // % subtraction domain: 1=magnitude, 2=power
	double e=1;      //  % gain exponent
	double am=3;     // % max oversubtraction factor
	double b=0.01;   //   % noise floor
	double al=-5;    //   % SNR for maximum a (set to Inf for fixed a)
	double ah=20;    //   % SNR for minimum a
	double bt=-1;    //   % suppress binary masking
	int ne=0;      //  % noise estimation: 0=min statistics, 1=MMSE [0]
	int mx=0;      //  % no input mixing
	double gh=1;   //     % maximum gain
	char tf='g';   //   % output the gain time-frequency plane by default
	double rf=0;

	// Parameter values from VQVAD
	qq_g      = 2;
	e      = 2;
	ne     = 1;
	am     = 10; //% allow aggressive oversubtraction

	// % derived algorithm constants
	float ni= (float) round(ti*fs);   // % frame increment in samples
	double tinc=ni/fs;       //   % true frame increment time

	// % calculate power spectrum in frames
	float no=(float) round(of);                  //                 % integer overlap factor
	float nf=ni*no;          // % fft length

	/* actual values:
	no = 2;
	nf = 16e-3 * 2;
	*/

	/*
	w=sqrt(hamming(nf+1))'; w(end)=[]; % for now always use sqrt hamming window
	w=w/sqrt(sum(w(1:ni:nf).^2));       % normalize to give overall gain of 1
	rfm='r';
	[y,tt]=enframe(s,w,ni,rfm);
	tt=tt/fs;                           % frame times
	yf=rfft(y,nf,2);
	*/

	final ArrayList<DataProcessor> pipeline = new ArrayList<DataProcessor>();
	pipeline.add(new BaseDataProcessor() {
		boolean sent = false;
		@Override
		public Data getData() throws DataProcessingException {
			if (!sent) {
				sent = true;
				return new DoubleData(s);
			}
			return null;
		}
	});
	pipeline.add(new RaisedCosineWindower(0.46,nf,ni));
	pipeline.add(new DiscreteFourierTransform());
	FrontEnd f = new FrontEnd(pipeline);
	List<DoubleData> yp_buffer = new ArrayList<DoubleData>();

	Data d;
	do {
		d = f.getData();

		if (d instanceof DoubleData) {
			DoubleData dd = (DoubleData) d;
			yp_buffer.add(dd);
		}

	} while(d != null);

	double[][] yp = new double[yp_buffer.size()][];

	// yp=yf.*conj(yf);        % power spectrum of input speech
	// Already a result of DFT().
	int ypi = 0;
	for (DoubleData y : yp_buffer) {
		yp[ypi] = y.getValues();
		ypi++;
	}

	// 	[nr,nf2]=size(yp);              % number of frames
	int nr = yp.length;
	int nf2 = yp[0].length;

	// ff=(0:nf2-1)*fs/nf;
	double ff[] = new double[nf2];
	for (int i=0; i < nf2; i++)
		ff[i] = i * fs / nf;

	// [dp,ze]=estnoiseg(yp,tinc,qp);	% estimate the noise using MMSE
	double[][] dp = estnoiseg(yp, tinc);

	// ssv=zeros(ni*(no-1),1);             % dummy saved overlap
	double[] ssv = new double[(int) (ni*(no-1))];
	for (int i=0; i < ssv.length; i++) ssv[i] = 0;


	// mz=yp==0;   %  mask for zero power time-frequency bins (unlikely)
	double[][] mz = new double[nr][nf2];
	for (int i=0; i < nr; i++) {
		for (int j=0; j < nf2; j++) {
			mz[i][j] = (yp[i][j] == 0) ? 1 : 0;
		}
	}

	// ypf=sum(yp,2); // row-wise sum
	double ypf[] = new double[nr];
	for (int i=0; i < nr; i++) {
		ypf[i] = 0;
		for (int j=0; j < nf2; j++) {
			ypf[i] += yp[i][j];
		}
	}

	//dpf=sum(dp,2); // row-wise sum
	double dpf[] = new double[dp.length];
	for (int i=0; i < nr; i++) {
		dpf[i] = 0;
		for (int j=0; j < dp[0].length; j++) {
			dpf[i] += dp[i][j];
		}
	}

	// mzf=dpf==0;     % zero noise frames = very high SNR
	double[] mzf = new double[dpf.length];
	for (int i=0; i < dpf.length; i++) {
		mzf[i] = (dpf[i] == 0) ? 1 : 0;
	}


	// af=1+(qq.am-1)*(min(max(10*log10(ypf./(dpf+mzf)),qq.al),qq.ah)-qq.ah)/(qq.al-qq.ah);
	double[] af_col = new double[ypf.length];
	for (int i=0; i < af_col.length; i++) {
		af_col[i] = 1 + (am-1) * (Math.min(Math.max(10*Math.log10(ypf[i] / (dpf[i]+mzf[i])), al), ah) - ah) / (al-ah);
	}

	// af(mzf)=1;      % fix the zero noise frames
	for (int i=0; i < af_col.length; i++) {
		af_col[i] = (mzf[i] == 1) ? 1 : af_col[i];
	}


	// % power domain subtraction
	// v=dp./(yp+mz);
	double[][] v = new double[nr][nf2];
	for (int i=0; i < nr; i++) {
		for (int j=0; j < nf2; j++) {
			v[i][j] = dp[i][j] / (yp[i][j] + mz[i][j]);
		}
	}

	double bf=b;

	// af =repmat(af,1,nf2);       % replicate frame oversubtraction factors for each frequency
	// column-repetition of af_col.
	//
	// af = | | | with | being af_col
	double[][] af = new double[af_col.length][nf2];
	for (int j=0; j < nf2; j++) {
		for (int i=0; i < af_col.length; i++) {
			af[i][j] = af_col[i];
		}
	}

	// mf=v>=(af+bf).^(-1);        % mask for noise floor limiting
	double[][] mf = new double[v.length][v[0].length];
	for (int i=0; i < v.length; i++) {
		for (int j=0; j < v[0].length; j++) {
			mf[i][j] = (v[i][j] >= 1/(af[i][j] + bf)) ? 1 : 0;
		}
	}

	// g=zeros(size(v));           % reserve space for gain matrix
	double[][] g = new double[v.length][v[0].length];
	// eg=qq.e/qq.g;               % gain exponent relative to subtraction domain
	double eg = e / qq_g;


	// % Normal case
	// g(mf)=min(bf*v(mf),gh);      % never give a gain > 1
	for (int i=0; i < g.length; i++) {
		for (int j=0; j < g[0].length; j++) {
			// TODO
		}
	}

	// g(~mf)=1-af(~mf).*v(~mf);


			/*
	if qq.bt>=0
		g=g>qq.bt;
	end
	g=qq.mx+(1-qq.mx)*g;   % mix in some of the input
	se=(irfft((yf.*g).',nf).').*repmat(w,nr,1);   % inverse dft and apply output window
	ss=zeros(ni*(nr+no-1),no);                      % space for overlapped output speech
	ss(1:ni*(no-1),end)=ssv;
	for i=1:no
		nm=nf*(1+floor((nr-i)/no));  % number of samples in this set
		ss(1+(i-1)*ni:nm+(i-1)*ni,i)=reshape(se(i:no:nr,:)',nm,1);
	end
	ss=sum(ss,2);

	ss=ss(1:length(s));             % trim to the correct length if not an exact number of frames
	*/
		return null;
	}

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		// TODO Auto-generated method stub

	}

}
