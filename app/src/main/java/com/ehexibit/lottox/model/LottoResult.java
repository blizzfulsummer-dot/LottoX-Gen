package com.ehexibit.lottox.model;

import java.util.*;
import java.util.stream.Collectors;

public class LottoResult {
    static int NUMBER = 0;

    private final List<Integer> IDs;          // Store LottoNumber IDs
    private final LottoNumber[] numbers;      // Original LottoNumbers
    private final String res;                 // String representation
    private String date;

    public LottoResult(LottoNumber[] numbers) {
        this.numbers = numbers;
        this.IDs = Arrays.stream(numbers)
                         .map(LottoNumber::getID)
                         .collect(Collectors.toList());

        // Build string representation
        this.res = Arrays.stream(numbers)
                         .map(n -> String.valueOf(n.getValue()))
                         .collect(Collectors.joining(" "));

        setBestPair();
        NUMBER++;
    }

    private void setBestPair() {
        for (int i = 0; i < numbers.length; i++) {
            for (int j = 0; j < numbers.length; j++) {
                if (i != j) {
                    numbers[i].setBestPairIDs(numbers[j]);
                }
            }
        }
    }

    public void setDate(String date) {
        this.date = date;
    }

    public List<Integer> getIDs() {
        return IDs;
    }

    public String getDate() {
        return date;
    }

    // ✅ Generic containsAll for any number of LottoNumbers
    public boolean containsAll(LottoNumber... nums) {
        List<Integer> check = Arrays.stream(nums)
                                    .map(LottoNumber::getID)
                                    .toList();
        return IDs.containsAll(check);
    }

    public void printResult() {
        IDs.forEach(id -> System.out.print(id < 10 ? "0" + id + " " : id + " "));
        System.out.println();
    }

    @Override
    public String toString() {
        return res;
    }
}
