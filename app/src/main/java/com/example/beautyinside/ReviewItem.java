package com.example.beautyinside;

public class ReviewItem {
    private String before;
    private String after;
    private String hospital;
    private String doctor;
    private String procedure;
    private float similarity;
    // 생성자 + getter 추가
    public ReviewItem(String hospital, String doctor, String procedure,
                      String before, String after, float similarity) {
        this.hospital = hospital;
        this.doctor = doctor;
        this.procedure = procedure;
        this.before = before;
        this.after = after;
        this.similarity = similarity;
    }
    public String getBefore() { return before; }
    public String getAfter() { return after; }
    public String getHospital() { return hospital; }
    public String getDoctor() { return doctor; }
    public String getProcedure() { return procedure; }
    public float getSimilarity() {
        return similarity;
    }

}
