/* Get the source code and data from here: http://projects.csail.mit.edu/jmwe/
* Download them, and put this file in src/edu.mit.mwe2011/wsd, next to the other experiment classes
* Then run it and pipe the output to a text file to generate the data */

package edu.mit.mwe2011;

import edu.mit.jmwe.data.IMWE;
import edu.mit.jmwe.data.IMarkedSentence;
import edu.mit.jmwe.data.IToken;
import edu.mit.jmwe.data.concordance.IConcordanceSentence;
import edu.mit.jmwe.data.concordance.TaggedConcordanceIterator;
import edu.mit.jmwe.harness.ConcordanceAnswerKey;
import edu.mit.jsemcor.main.IConcordanceSet;

import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;

public class PrintData extends AbstractExperiment implements Runnable {

	public static void main(String[] args) {
		new PrintData().run();
	}

	public void run() {
		try {
			
			File semcorDir = getSemcorDirectory();
			File taggedSemcorFile = getTaggedSemcorData();
			IConcordanceSet semcor = getSemcor(semcorDir);

			// jMWE calculations
			printData(taggedSemcorFile, semcor);
			
		} catch (Throwable e) {
			if(e instanceof RuntimeException){
				throw (RuntimeException)e;
			} else {
				throw new RuntimeException(e);
			}
		} 
	}

	protected void printData(File taggedSemcorFile, IConcordanceSet semcor) throws IOException {
		Iterator<IConcordanceSentence> itr = new TaggedConcordanceIterator(taggedSemcorFile);
		ConcordanceAnswerKey key = new ConcordanceAnswerKey(semcor);

		while(itr.hasNext()) {
			IMarkedSentence sent = itr.next();
			List<IMWE<IToken>> answerMWEs = key.getAnswers(sent);
			HashMap<String, String> outputMap = new HashMap<>();
			System.out.println(sent.toString());
			for (IMWE<IToken> mwe: answerMWEs) {
				System.out.println(mwe.toString());
			}
			System.out.println("--------");
		}
	}	

}
