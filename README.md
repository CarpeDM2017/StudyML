Welcome to Carpe DM 2017!!
==========================

### 2017.09.21

1. 활동내용 관련 workout 폴더를 추가했습니다. 새로 파일을 업로드하기 전에 우선 pull 부탁드립니다.

2. 1주차 활동내용 jupyter notebook을 추가하였습니다. 9/24 일요일까지 꼭 작성해주세요 :)


### 2017.09.10

1. 모두 github 아이디를 만들어주세요.
   가입하신 뒤 사용자명과 이메일을 카톡방에 올려주세요.


2. git을 설치해주세요.
   <pre>https://git-scm.com/downloads
   https://git-scm.com/book/en/v2/Getting-Started-First-Time-Git-Setup</pre>


3. git을 설치한 뒤에 터미널 / 또는 커맨드 프롬프트에서 다음을 실행해 유저명과 이메일을 설정해주세요.
   <pre>git config --global user.name "홍길동"
   git config --global user.email "이메일@어디.컴"</pre>

   다음 커맨드를 입력해 본인이 기입한 내용이 잘 들어갔는지 확인해주세요.
   <pre>git config --global user.name
   git config --global user.email</pre>


4. 이제 Repository에서 내용을 긁어옵시다. 
   원하는 경로로 (바탕화면 등) 터미널 / 커맨드 프롬프트를 통해 이동한 뒤 이 커맨드를 입력하세요.
   <pre>git clone https://github.com/CarpeDM2017/StudyML</pre>

   Windows 커맨드 프롬프트 명령어
   
   http://library1008.tistory.com/42

   리눅스 터미널 명령어
   
   http://www.mireene.com/webimg/linux_tip1.htm

   Mac 터미널 명령어
   
   https://github.com/tadkim/infra/wiki/Mac-::-%ED%84%B0%EB%AF%B8%EB%84%90-%EB%AA%85%EB%A0%B9%EC%96%B4 

   깃 명령어
   
   https://git-scm.com/docs
   
   꼭 필요한 명령어는 [add, status, commit, push, pull], 커밋이 꼬이는 경우 유용하게 사용할 수 있는 명령어들은 [stash, rebase, reset], 버전 및 브랜치 관리에 필요한 명령어들은 [merge, branch, checkout, fetch] 등이 있습니다.


5. clone을 통해 생성한 StudyML 폴더로 들어가서 KYH 폴더를 복사하여, 본인의 이니셜로 수정해 추가해주세요. 폴더 안에 있는 README 파일을 열어, 본인의 이름으로 수정해주세요.

   마크다운 문서 사용법
   
   https://www.3rabbitz.com/markdown_guide


   이후 다시 터미널 / 커맨드 프롬프트를 통해 다음의 명령어를 순서대로 실행합니다.
   <pre>git add .
   git commit -m "added [본인 이니셜] folder"
   git push</pre>