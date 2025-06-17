package com.example.beautyinside;

import android.os.Bundle;
import androidx.fragment.app.Fragment;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;
import com.example.beautyinside.R;
import com.example.beautyinside.adapter.DoctorAdapter;
import com.example.beautyinside.model.Hospital;
import java.util.ArrayList;
import java.util.List;

public class DoctorFragment extends Fragment {

    public DoctorFragment() {}

    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container,
                             Bundle savedInstanceState) {

        View view = inflater.inflate(R.layout.fragment_doctor, container, false);

        RecyclerView recyclerView = view.findViewById(R.id.doctor_recycler_view);
        recyclerView.setLayoutManager(new LinearLayoutManager(getContext()));

        List<Hospital> hospitalList = new ArrayList<>();
        hospitalList.add(new Hospital("팝 성형외과", "믿을 수 있는 결과, 팝에서 시작하세요", "pop"));
        hospitalList.add(new Hospital("티에스 성형외과", "자연스러움을 추구하는 병원", "ts"));
        hospitalList.add(new Hospital("마인드 성형외과", "당신의 마음까지 아름답게", "mind"));
        hospitalList.add(new Hospital("아몬드 성형외과", "환한 눈빛을 위한 선택", "amond"));

        hospitalList.add(new Hospital("나나 성형외과", "나를 위한 특별한 변화", "nana"));
        hospitalList.add(new Hospital("드레스 성형외과", "드레스를 위한 자신감", "dress"));
        hospitalList.add(new Hospital("디엠 성형외과", "디테일을 완성하는 병원", "dm"));
        hospitalList.add(new Hospital("루호 성형외과", "당신만의 아름다움 루호", "ruho"));

        hospitalList.add(new Hospital("옐로우 성형외과", "활짝 핀 미소, 옐로우", "yellow"));
        hospitalList.add(new Hospital("히트 성형외과", "요즘 핫한 그곳!", "hit"));
        hospitalList.add(new Hospital("마블 성형외과", "마블처럼 다채로운 아름다움", "marble"));
        hospitalList.add(new Hospital("온에어 성형외과", "방송처럼 완벽한 스타일", "onair"));
        hospitalList.add(new Hospital("클래시 성형외과", "우아함의 정석, 클래시", "clasy"));

        hospitalList.add(new Hospital("유노 성형외과", "너만을 위한 아름다움", "youno"));
        hospitalList.add(new Hospital("메이드영 성형외과", "젊음을 다시 디자인하다", "made"));
        hospitalList.add(new Hospital("바바 성형외과", "믿고 보는 바바", "baba"));

        hospitalList.add(new Hospital("땡큐 성형외과", "고객 감동 1위", "thankyou"));
        hospitalList.add(new Hospital("원더풀 성형외과", "결과가 원더풀!", "wonder"));
        hospitalList.add(new Hospital("이에스 성형외과", "E.S = Elegant & Safe", "es"));
        hospitalList.add(new Hospital("일미리 성형외과", "하루만에 변화, 일미리", "ml"));

        DoctorAdapter adapter = new DoctorAdapter(getContext(), hospitalList);
        recyclerView.setAdapter(adapter);

        return view;
    }
}
