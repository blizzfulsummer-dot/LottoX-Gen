package com.ehexibit.lottox.model;

import java.util.*;

public class LottoNumber {
    static int COUNT = 0; // count the LottoNumber object created
    private Lotto draw;
    private int ID;              // If Number 1, ID is 1
    private String value;        // "01", "02", ...
    private boolean selected;
    private LottoNumber[] numbers;
    private int frequency = 0;
    private boolean suggested = false;

    // Instead of int[][], use a Map to track best pair frequencies
    private Map<LottoNumber, Integer> bestPairs = new HashMap<>();

    // Constructor
    public LottoNumber() {
        init();
        COUNT++;
    }

    // Initial state
    public void init() {
        selected = false;
        value = "00";
    }

    public void setSuggested(boolean b) { suggested = b; }
    public boolean getSuggested() { return suggested; }

    public void setDraw(Lotto draw) { this.draw = draw; }
    public Lotto getDraw() { return draw; }

    public void selected(boolean b) { selected = b; }

    public void setID(int ID) {
        this.ID = ID;
        value = String.format("%02d", ID);
    }

    public String getValue() { return value; }
    public boolean isSelected() { return selected; }
    public int getID() { return ID; }

    public void setNumbers(LottoNumber[] n) {
        this.numbers = n;
        bestPairs.clear(); // reset pair stats
        for (LottoNumber num : n) {
            if (num != this) {
                bestPairs.put(num, 0);
            }
        }
    }

    // Increment frequency with another LottoNumber
    public void setBestPairIDs(LottoNumber a) {
        bestPairs.put(a, bestPairs.getOrDefault(a, 0) + 1);
        frequency++;
    }

    public int getFrequency() { return frequency; }

    // Get the best (most frequent) partner that isn’t selected
    public LottoNumber getBestPair() {
        return bestPairs.entrySet().stream()
                .filter(e -> !e.getKey().isSelected() && e.getKey() != this)
                .max(Map.Entry.comparingByValue())
                .map(Map.Entry::getKey)
                .orElse(null);
    }

    // Find a common partner between this and another number
    public LottoNumber getCommon(LottoNumber other) {
        LottoNumber best = getBestPair();
        if (best == null) return null;

        for (Map.Entry<LottoNumber, Integer> entry : bestPairs.entrySet()) {
            LottoNumber candidate = entry.getKey();
            if (!candidate.isSelected() && candidate != this) {
                if (candidate.equals(other.getBestPair())) {
                    candidate.selected(true);
                    return candidate;
                }
            }
        }

        // fallback: return current best
        best.selected(true);
        return best;
    }
}
