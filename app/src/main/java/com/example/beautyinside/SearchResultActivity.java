package com.example.beautyinside;

import android.os.Bundle;
import android.text.Editable;
import android.text.TextWatcher;
import android.widget.EditText;
import androidx.appcompat.app.AppCompatActivity;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import com.example.beautyinside.adapter.HospitalAdapter;

import java.util.ArrayList;
import java.util.List;

public class SearchResultActivity extends AppCompatActivity {

    private List<HospitalData> fullHospitalList = new ArrayList<>();     // 전체 병원 목록
    private List<HospitalData> filteredHospitalList = new ArrayList<>(); // 검색 필터 결과
    private HospitalAdapter adapter;
    private RecyclerView recyclerView;
    private EditText searchInput;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_search_result);

        recyclerView = findViewById(R.id.recycler_search_results);
        searchInput = findViewById(R.id.search_edit_text);

        recyclerView.setLayoutManager(new LinearLayoutManager(this));

        // 예시로 static list 사용 - 실제 앱에서는 DB 또는 Intent 등으로 받아오기
        fullHospitalList = getHospitalList();
        filteredHospitalList.addAll(fullHospitalList);

        adapter = new HospitalAdapter(this,filteredHospitalList);
        recyclerView.setAdapter(adapter);

        searchInput.addTextChangedListener(new TextWatcher() {
            @Override
            public void beforeTextChanged(CharSequence s, int start, int count, int after) {}

            @Override
            public void onTextChanged(CharSequence s, int start, int before, int count) {
                filterHospitalList(s.toString());
            }

            @Override
            public void afterTextChanged(Editable s) {}
        });
    }

    private void filterHospitalList(String keyword) {
        filteredHospitalList.clear();

        if (keyword == null || keyword.trim().isEmpty()) {
            filteredHospitalList.addAll(fullHospitalList);
        } else {
            for (HospitalData hospital : fullHospitalList) {
                if (hospital.getName().contains(keyword)) {
                    filteredHospitalList.add(hospital);
                }
            }
        }

        adapter.notifyDataSetChanged();
    }

    // 테스트용 병원 목록 (실제 프로젝트에선 삭제하고 데이터 받아오는 방식으로 대체)
    private List<HospitalData> getHospitalList() {
        List<HospitalData> list = new ArrayList<>();
        list.add(new HospitalData("팝성형외과" , R.drawable.pop_1));
        list.add(new HospitalData("아몬드성형외과", R.drawable.amond_1));
        list.add(new HospitalData("티에스성형외과", R.drawable.ts_1));
        list.add(new HospitalData("바바성형외과" , R.drawable.baba_1));
        list.add(new HospitalData("클래시성형외과", R.drawable.clasy_1));
        list.add(new HospitalData("나나성형외과", R.drawable.nana_1));
        list.add(new HospitalData("마인드외과" , R.drawable.mind_1));
        list.add(new HospitalData("일미리성형외과", R.drawable.ml_1));
        list.add(new HospitalData("메이드영성형외과", R.drawable.made_1));
        list.add(new HospitalData("유노성형외과" , R.drawable.youno_1));
        list.add(new HospitalData("루호성형외과", R.drawable.ruho_1));
        list.add(new HospitalData("온에어성형외과", R.drawable.onair_1));
        list.add(new HospitalData("옐로우성형외과" , R.drawable.yellow_1));
        list.add(new HospitalData("원더풀성형외과", R.drawable.wonder_1));
        list.add(new HospitalData("땡큐성형외과", R.drawable.thankyou_1));
        list.add(new HospitalData("마블성형외과" , R.drawable.marble_1));
        list.add(new HospitalData("드레스성형외과", R.drawable.dress_1));
        list.add(new HospitalData("디엠성형외과", R.drawable.dm_1));
        list.add(new HospitalData("히트성형외과" , R.drawable.hit_1));
        list.add(new HospitalData("이에스성형외과", R.drawable.es_1));


        return list;
    }
}