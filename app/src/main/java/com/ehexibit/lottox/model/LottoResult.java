package com.ehexibit.lottox.model;

public class LottoResult{
	static int NUMBER=0;
	private int ID[];
	private LottoNumber[] numbers;
	private String res;
	
	String date;
	
	public LottoResult(LottoNumber[] numbers){
	 this.numbers = numbers;
	 res = "";
	 ID = new int[numbers.length];
	 for(int i=0; i<numbers.length; i++){
		 res += ""+numbers[i].getValue()+" ";
		 ID[i]=numbers[i].getID();

		 
	 }
	 
	 setBestPair();
	 NUMBER++;
	 
	}
	public void setBestPair() {
    	for (int i = 0; i < numbers.length; i++) {
        	for (int j = i + 1; j < numbers.length; j++) {
            	// set both ways at the same time
            	numbers[i].setBestPairIDs(numbers[j]);
            	numbers[j].setBestPairIDs(numbers[i]);
        	}
    	}
	}

	//Date to be implement later
	public void setDate(String date){
		this.date = date;
	}	
	public int[] getIDs(){
		return ID;
	}
	public String getDate(){
		return date;
	}
	public boolean contains(LottoNumber a){
		
		for(int n:ID){
			if(n==a.getID()) return true;
		}
		return false;
	}
	
	public boolean contains(LottoNumber a,LottoNumber b){
		
		for(int i=0; i<numbers.length; i++){
			for(int j=i+1; j<numbers.length; j++){
		 		if(ID[i]==a.getID()&&ID[j]==b.getID()
		 	  	||ID[j]==a.getID()&&ID[i]==b.getID())return true;
			}
		}
		return false;
	}
	
	public boolean contains(LottoNumber a, LottoNumber b, LottoNumber c){
		if(contains(a)&&contains(b,c))
						return true;
		return false;
	}
	
	public boolean contains(LottoNumber a, LottoNumber b, LottoNumber c, LottoNumber d){
		if(contains(a,b)&&contains(c,d))
						return true;
		return false;
	}
	public void printResult(){
		for(int a:ID)
		System.out.print(a<10?"0"+a+" ":a+" ");
		System.out.println();
	}

	public String toString(){
		
		return res;
	}

}
