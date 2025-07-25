% 文件名：scuthesis2020.cls
% 文件描述：以 ctexbook 文档类为基础，定义四川大学学位论文文档类
% 作者：LegendaryLeo
% 修改人：Kevin T. Lee [hello@lidengju.com]
% 修改日期：2020年6月1日
\NeedsTeXFormat{LaTeX2e}[1995/12/01]
\ProvidesClass{./Template/scuthesis2020}[2020/06/01]

% 加载key value options
\RequirePackage{xkeyval}

\def\approval{0}
\def\colortitle{0}
\def\secret{0}
\DeclareOption{doctor}{\def\thesistype{0}}
\DeclareOption{master}{\def\thesistype{1}}
\DeclareOption{approval}{\def\approval{1}}
\DeclareOption{color}{\def\colortitle{1}}
\DeclareOption{secret}{\def\secret{1}}
\DeclareOption{academic}{\def\degreetype{0}}
\DeclareOption{professional}{\def\degreetype{1}}

\newcommand{\kkfont}{}
\DeclareOptionX{kfont}{\renewcommand{\kkfont}{#1}}
% \DeclareOptionX*{\PackageWarning{scuthesis2020}{`\CurrentOption' ignored}}% For unknown options
\ExecuteOptionsX{kfont=fandol}% Preset keys, '' being the default here
\ProcessOptionsX\relax

% 将文档类其他参数全部传递给 ctexbook 文档类。
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{ctexbook}}
\ProcessOptions\relax
% 此文档类以 ctexbook 为基础，设定页面尺寸为 A4，默认字号为小四号。
% CJKfntef 宏包提供了 \maketitle 中用到的 \CJKunderline 命令。
\LoadClass[a4paper,UTF8,twoside, openright, zihao=-4,fontset=\kkfont]{ctexbook}

% 解图表目录引用编号顺序问题
\RequirePackage{notoccite}
% 加入hyperref超链接参考
\RequirePackage{hyperref}
% fancyhdr 宏包提供了方便使用的命令用于设置页眉、页脚。
\RequirePackage{fancyhdr}
% 加入判断支持
\RequirePackage{ifthen}
% 此宏包提供图形支持。
\RequirePackage{graphicx}
% 颜色
\RequirePackage{xcolor}
% lipsum
\RequirePackage{lipsum}
% 定义表格中的多行单元
\RequirePackage{multirow}
\newcommand{\tabincell}[2]{\begin{tabular}{@{}#1@{}}#2\end{tabular}}
% 引入landscape排版
\RequirePackage{lscape}
% 引入算法包
\RequirePackage{algorithm}
\RequirePackage{algorithmic}
\floatname{algorithm}{算法}  
\renewcommand{\algorithmicrequire}{\textbf{输入:}}  
\renewcommand{\algorithmicensure}{\textbf{输出:}}

% 此包用于双语图表标题
\RequirePackage[list=off]{bicaption}
\RequirePackage{subcaption}
\captionsetup{font={bf},bi-slc=0}
\captionsetup[figure][bi-first]{name=图,labelsep=quad,labelfont={rm,bf,small},font={rm,bf,small}}
\captionsetup[figure][bi-second]{name=Figure,labelsep=quad,labelfont={rm,bf,small},font={rm,bf,small}}
\captionsetup[table][bi-first]{name=表,labelsep=quad,labelfont={rm,bf,small},font={rm,bf,small}}
\captionsetup[table][bi-second]{name=Table,labelsep=quad,labelfont={rm,bf,small},font={rm,bf,small}}
% 修改图题和正文间距
% \setlength{\belowcaptionskip}{-10pt}
%设置字体
\RequirePackage{xeCJK}
% 手动载入xeCJKfntef包，用于兼容2.5以上的ctex版本
\RequirePackage{xeCJKfntef}

%使用geometry调整边距
\RequirePackage{geometry}
\geometry{a4paper,papersize={21cm,29.7cm}}
\geometry{left=2.5cm,right=2.5cm,top=2.75cm,bottom=3cm,headsep=0.32cm,footskip=1cm}
% bottom为2.5 + 0.5cm，考虑到footer的高度，这样与word一致
% \setlength{\topskip}{0.3cm}
% \renewcommand{\baselinestretch}{1.5}
\setlength{\baselineskip}{20pt}
% 这些命令用于设定各项文档信息的具体内容。
\newcommand\unitid[1]{\def\@unitid{#1}}
\newcommand\STUnumber[1]{\def\@STUnumber{#1}}
\newcommand\securityClassification[1]{\def\@securityClassification{#1}}
\newcommand\securityYear[1]{\def\@securityYear{#1}}
\newcommand\CoverTitle[1]{\def\@CoverTitle{#1}}
\newcommand\CoverSubTitle[1]{\def\@CoverSubTitle{#1}}
\renewcommand{\title}[1]{\def\@title{#1}}
\newcommand\ENGtitle[1]{\def\@ENGtitle{#1}}
\newcommand\school[1]{\def\@school{#1}}
\newcommand\ENGschool[1]{\def\@ENGschool{#1}}
\newcommand\ENGauthor[1]{\def\@ENGauthor{#1}}
\newcommand\supervisor[1]{\def\@supervisor{#1}}
\newcommand\ENGsupervisor[1]{\def\@ENGsupervisor{#1}}
\newcommand\degreeclass[1]{\def\@degreeclass{#1}}
\newcommand\ENGdegreeclass[1]{\def\@ENGdegreeclass{#1}}
\newcommand\major[1]{\def\@major{#1}}
\newcommand\ENGmajor[1]{\def\@ENGmajor{#1}}
\newcommand\hasmajor[1]{\def\@hasmajor{#1}}
\newcommand\defensedate[1]{\def\@defensedate{#1}}
\newcommand\grantdate[1]{\def\@grantdate{#1}}
\newcommand\accomplishdate[1]{\def\@accomplishdate{#1}}
\newcommand\statementdate[1]{\def\@statementdate{#1}}
\newcommand\direction[1]{\def\@direction{#1}}
\newcommand\ENGdirection[1]{\def\@ENGdirection{#1}}
\newcommand\keywords[1]{\def\@keywords{#1}}
\newcommand\ENGkeywords[1]{\def\@ENGkeywords{#1}}
\newcommand\university[1]{\def\@university{#1}}
\newcommand\ENGuniversity[1]{\def\@ENGuniversity{#1}}

% 判断论文类别，定义不同论文名称
\ifthenelse{\equal{\thesistype}{0}}{
\def\thesisname{博士研究生学位论文}
\def\CHSauthortype{研究生}
\def\ENGauthortype{Postgraduate}
\def\TypeAbbr{PhD Degree}
}{}
\ifthenelse{\equal{\thesistype}{1}}{
\def\thesisname{硕士研究生学位论文}
\def\CHSauthortype{研究生}
\def\ENGauthortype{Postgraduate}
\def\TypeAbbr{MSc Degree}
}{}
\ifthenelse{\equal{\degreetype}{0}}{
\def\proffieldname{学科专业}
\def\degreename{学术学位}
}{}
\ifthenelse{\equal{\degreetype}{1}}{
\def\proffieldname{领域名称}
\def\degreename{专业学位}
}{}

%定义字体格式
\newcommand{\fontq}{
	\zihao{-3}\heiti
}
\newcommand{\fonta}{
	\zihao{3}\songti
}
\newcommand{\makesecret}{
	\ifthenelse{\equal{\secret}{1}}{
		\\
		{\makebox[2.2cm][s]{\textbf{密级：}}}       & \hspace{-10pt}\textbf{\@securityClassification\quad\@securityYear 年}
	}{}

}

\newcommand{\makescucoverhead}
{
\ifthenelse{\equal{\approval}{0}}{
		\zihao{-4}\songti
		\begin{tabular}{cl}
			{\makebox[2.2cm][s]{\textbf{单位代码：}}}  & \hspace{-10pt}\textbf{\@unitid}\\
			{\makebox[2.2cm][s]{\textbf{学号：}}}       & \hspace{-10pt}\textbf{\@STUnumber}
			\makesecret
		\end{tabular}
	}{}
\ifthenelse{\equal{\approval}{1}}{
		\zihao{-4}\songti
		\begin{tabular}{cl}
			{\makebox[2.2cm][s]{\textbf{单位代码：}}}  & \hspace{-10pt}\textbf{\@unitid}\\
			{\makebox[2.2cm][s]{\textbf{送审编号：}}}       & \hspace{-10pt}\textbf{\@STUnumber}
			\makesecret
		\end{tabular}
		
	}{}
\ \\[28pt]
}

\newcommand{\makescucoverimage}
{
\ifthenelse{\equal{\colortitle}{0}}{
	\includegraphics[width=10cm]{./Template/Components/Images/SCU_TITLE_BW}
	\ \\[0cm]
}{}
\ifthenelse{\equal{\colortitle}{1}}{
	\includegraphics[width=10cm]{./Template/Components/Images/SCU_TITLE}
	\ \\[0.5cm]
}{}
{\makebox[9cm][s]{\zihao{1}\heiti\textbf{\thesisname}}}
\\[18pt]
{\makebox[5cm][c]{\zihao{3}\songti\text{（}\degreename\text{）}}}
\\[47pt]
}

\newcommand{\makescucovertitle}
{
\begin{tabular}{lc}
	{\heiti\zihao{3}\makebox[1.25cm][s]{\textbf{题目：}}}  & \hspace{-10pt}\fillinblank{10.5cm}{\heiti\zihao{3}\textbf{\@CoverTitle}}
	\\[25pt]
	\ifthenelse{\equal{\@CoverSubTitle}{}}{}{
	{\makebox[2cm][s]{}}       & \hspace{-10pt}\fillinblank{10.5cm}{\heiti\zihao{3}\textbf{\@CoverSubTitle}}
	}
\end{tabular}
\\[45pt]
}

\newcommand{\makescucovercontent}
{
%非送审版本
\ifthenelse{\equal{\approval}{0}}{
	
	\begin{tabular}{cc}
		{\hspace{-0.7cm}\fontq\makebox[2.8cm][l]{培养单位：}}  & \hspace{-0.7cm}\fonta\fillinblank{8cm}{\@school}
		\\[17pt]
		{\hspace{-0.7cm}\fontq\makebox[2.8cm][l]{作者姓名：}}  & \hspace{-0.7cm}\fonta\fillinblank{8cm}{\@author}
		\\[17pt]
		{\hspace{-0.7cm}\fontq\makebox[2.8cm][l]{指导教师：}}       & \hspace{-0.7cm}\fonta\fillinblank{8cm}{\@supervisor}
		\\[17pt]
		{\hspace{-0.7cm}\fontq\makebox[2.8cm][l]{学位类别：}}   & \hspace{-0.7cm}\fonta\fillinblank{8cm}{\@degreeclass}
		\\[17pt]
		{\hspace{-0.7cm}\fontq\makebox[2.8cm][l]{\proffieldname ：}} & \hspace{-0.7cm}\fonta\fillinblank{8cm}{\@major}
	\end{tabular}
	\\[48pt]
	\zihao{3}\songti
	\begin{tabular}{cc}
		{\makebox[3cm][c]{\text{论文答辩时间：}}} & {\@defensedate} \\[0.4cm]
		{\makebox[3cm][c]{\text{学位授予时间：}}} & {\@grantdate}
	\end{tabular}
	% {\makebox[13cm][c]{\text{论文答辩时间：}\@defensedate}}
	% \\[12pt]
	% {\makebox[13cm][c]{\text{学位授予时间：}\@grantdate}}
	}{}
%送审版本
\ifthenelse{\equal{\approval}{1}}{
	\zihao{-3}\kaishu
	\begin{tabular}{cl}
		{\hspace{-0.7cm}\fontq\makebox[2.8cm][l]{学位类别：}}   & \hspace{-0.7cm}\fonta\fillinblank{8cm}{\@degreeclass}
		\\[17pt]
		{\hspace{-0.7cm}\fontq\makebox[2.8cm][l]{\proffieldname ：}} & \hspace{-0.7cm}\fonta\fillinblank{8cm}{\@major}
	\end{tabular}
	\\[70pt]
	\zihao{3}\songti
	\begin{tabular}{c}
		{\makebox[4cm][c]{\text{学位论文完成时间}}}\\[0.5cm]
		{\makebox[4cm][c]{\text{\@accomplishdate}}}
	\end{tabular}
	\\[30pt]
	{
		\zihao{5}
		\songti
		\begin{tabular}{cl}
			{注意事项：} & \multicolumn{1}{c}{按照双盲评阅要求，送审论文中不包含声明、致谢以及攻读学位期间取得的研究成果，} \\
			\textbf{}      & 所有与作者本人以及导师相关的信息，均须做匿名处理。
		\end{tabular}
	}
	}{}
}

\newcommand{\makekeywords}{
\vspace{1em}\par
\textbf{关键词：}\@keywords
\addcontentsline{toc}{chapter}{摘要}
}

\newcommand{\makeENGkeywords}{
\vspace{1em}\par
\textbf{Keywords：}\@ENGkeywords
\addcontentsline{toc}{chapter}{Abstract}
}

% 设定页眉、页脚样式。
% \fancypagestyle{plain}{\fancyhf{}\renewcommand{\headrulewidth}{0.8pt}}
\newcommand{\makefancyhdr}{
\fancypagestyle{plain}{\pagestyle{fancy}}
\pagestyle{fancy}
\fancyhf{}
\renewcommand{\headrulewidth}{0.8pt}
\fancyhead[CE]{\zihao{-5}\songti
	% The thesis title for even page
	\@title}

\renewcommand{\headrulewidth}{0.8pt}
\fancyhead[CO]{\zihao{-5}\songti
	% The chapter title for odd page
	\leftmark}
\fancyfoot[CO,CE]{\zihao{-5}\thepage}
}

\newcommand{\makefancyhf}[1]{
\fancypagestyle{plain}{\pagestyle{fancy}}
\pagestyle{fancy}
\fancyhf{}
\renewcommand{\headrulewidth}{0.8pt}
\fancyhead[CE]{\zihao{-5}\songti
	% The chapter title for even page
	\@title}

\renewcommand{\headrulewidth}{0.8pt}
\fancyhead[CO]{\zihao{-5}\songti
	% The chapter title for odd page
	#1}
\fancyfoot[CO,CE]{\zihao{-5}\thepage}
}

% 使用epstopdf宏包重定向eps输出路径，避免图片错误
\RequirePackage[outdir=./]{epstopdf}

% 使用hyperref宏包去除索引上的颜色标记
\RequirePackage{hyperref}
\hypersetup{pdfborder=0 0 0}
% 使用tocloft宏包修改目录样式
\RequirePackage{tocloft}
% 目录中不显示目录项、图表目录项
\RequirePackage[nottoc,notlof,notlot]{tocbibind}

% 定义 \incite 命令，用于如“文献[1]”这样的引用
\newcommand*{\incite}{}
\DeclareRobustCommand*{\incite}[2][]{
	\begingroup
	% \romannumeral-`\x
	\setcitestyle{numbers}
	\hspace{-1em}~\cite[#1]{#2}\hspace{-0.3em}
	\endgroup
}

% GBT7714-2015引用格式
\RequirePackage[sort&compress]{gbt7714}

% 使用natbib宏包修改参考文献段间距
\RequirePackage{natbib}
\setlength{\bibsep}{0pt}

%定义破折号
\newcommand{\cdash}[0]{\text{\ ——\ }}
% 公式编号格式(chapter-equation)
\renewcommand{\theequation}{\arabic{chapter}-\arabic{equation}}
% 支持自定义列表环境
\RequirePackage{enumitem}
\renewcommand\labelitemi{$\bullet$}
\setlist[itemize]{itemsep=2pt,parsep=0pt,topsep=0pt}
% \setlist[enumerate]{itemsep=0pt,parsep=0pt,topsep=0pt}
% \newenvironment{itemize*}%
%   {\begin{itemize}[itemsep=20pt,parsep=20pt]}%
%   {\end{itemize}}
% 支持ams数学宏包
\RequirePackage{amsfonts}
\RequirePackage{amsthm}
\RequirePackage{mathtools}
\RequirePackage{amssymb}

% 支持float图片环境
\RequirePackage{float}
% 三线表支持
\RequirePackage{booktabs}

% 设置标题深度为4
\setcounter{secnumdepth}{4}
% CTEX设置各级标题对齐方式，字号，字体及前后段间距
\renewcommand{\theparagraph}{(\arabic{paragraph})}
\ctexset{
	% “章节”标题，三号黑体，居中对齐，段前为24pt，段后18pt
	chapter= {
		format+={\rm\zihao{3}\heiti\bfseries},
		number={\arabic{chapter}},
		numberformat={\bfseries},
		% 设置序号和各级标题之间为一个空格，这里范例和写作说明不一致
		aftername={\hspace{\ccwd}},
		beforeskip={0pt},
		afterskip={18pt}
	},
	% “节”标题，四号黑体，左对齐，段前24pt，段后16pt
	section={
		format+={\rm\zihao{4}\heiti\bfseries\flushleft},
		% 设置序号和各级标题之间为两个空格，这里范例和写作说明不一致
		aftername={\hspace{\ccwd}},
		beforeskip={24pt},
		afterskip={6pt}
	},
	% “小节”标题，小四号黑体，左对齐，段前12pt，段后6pt
	subsection={
		format+={\zihao{-4}\rm\heiti\flushleft},
		% 设置序号和各级标题之间为一个空格，这里范例和写作说明不一致
		numberformat={\sffamily},
		aftername={\hspace{\ccwd}},
		beforeskip={12pt},
		afterskip={6pt}
	},
	% “次小节”标题，小四号楷体，左对齐，段前12pt，段后6pt
	subsubsection={
		format+={\zihao{-4}\rm\kaishu\flushleft},
		numberformat={\sffamily},
		% 设置序号和各级标题之间为一个空格，这里范例和写作说明不一致
		aftername={\hspace{\ccwd}},
		beforeskip={12pt},
		afterskip={6pt}
	},
	% “段”标题，小四号楷体，左对齐，前后段间距为0pt
	paragraph={
		name={,},
		number={\theparagraph},
		format+={\zihao{-4}\rm\kaishu\flushleft},
		beforeskip={0pt},
		afterskip={0pt}
	}
}

% 定义\makechaptertitlecenter命令，可使命令后章节标题居中
\newcommand{\makechaptertitlecenter}{\ctexset{
		chapter={
			format+={\center}
		}
	}}
% 定义\makechaptertitleleft命令，可使命令后章节标题居左
\newcommand{\makechaptertitleleft}{\ctexset{
		chapter={
			format+={\flushleft}
		}
	}}


% 定义\fillinblank命令，用下划线填充指定的空间。
\newcommand{\fillinblank}[2]{
\CJKunderline[thickness=0.8pt,depth=0.5em]{\makebox[#1]{#2}}
}
% 定义\putonbottom命令，将对象放到页面底部。
\newcommand{\putonbottom}[1]{
\begin{figure}[b!]%
	\begin{minipage}{\textwidth}
		{#1}
	\end{minipage}
\end{figure}
}

% 定义目录标题格式，三号黑体居中，段前后距10pt
\renewcommand\contentsname{目\hspace{1em}录}
\renewcommand{\cfttoctitlefont}{\hfill\zihao{3}\rm\heiti\bfseries}
\renewcommand{\cftchapfont}{\zihao{-4}\rm\heiti}
\renewcommand{\cftsecfont}{\zihao{-4}\rm\songti}
\renewcommand{\cftaftertoctitle}{\hfill}
%此处魔改目录标题前间距
\renewcommand{\cftbeforetoctitleskip}{8pt}
\renewcommand{\cftaftertoctitleskip}{10pt}
% 定义引导线格式，为点，点距0.5，章节标题也使用引导线
\renewcommand{\cftdot}{.}
\renewcommand{\cftdotsep}{0.5}
\renewcommand{\cftchapdotsep}{\cftdotsep}
% 定义章节目录项前后距
\renewcommand{\cftbeforechapskip}{6pt}

% 定义\maketoc命令，将自动生成符合要求的目录。
\newcommand{\maketoc}{
\makefancyhf{目录}
\cleardoublepage
\tableofcontents
\thispagestyle{fancy}
\makefancyhdr
}

% 定义图表目录标题格式，三号黑体居中，段前后距10pt
\renewcommand\listfigurename{插图目录}
\renewcommand\listtablename{表格目录}
\renewcommand{\cftloftitlefont}{\hfill\zihao{3}\rm\heiti\bfseries}
\renewcommand{\cftlottitlefont}{\hfill\zihao{3}\rm\heiti\bfseries}
\renewcommand{\cftafterloftitle}{\hfill}
\renewcommand{\cftafterlottitle}{\hfill}
%此处魔改目录标题前间距
\renewcommand{\cftbeforeloftitleskip}{8pt}
\renewcommand{\cftbeforelottitleskip}{8pt}
\renewcommand{\cftafterloftitleskip}{10pt}
\renewcommand{\cftafterlottitleskip}{10pt}

\newcommand{\loflabel}{图.}
\newcommand{\lotlabel}{表.}
\newlength{\mylen}
\renewcommand{\cftfigpresnum}{\figurename\enspace}
\settowidth{\mylen}{\cftfigpresnum\cftfigaftersnum}
\addtolength{\cftfignumwidth}{\mylen}

\renewcommand{\cfttabpresnum}{\tablename\enspace}
\settowidth{\mylen}{\cfttabpresnum\cfttabaftersnum}
\addtolength{\cfttabnumwidth}{\mylen}
% 生成图表目录
\newcommand{\makefigtablist}{
		\cleardoublepage
		\listoffigures
		\cleardoublepage
		\listoftables
}

% 设定封面页格式。
\renewcommand\maketitle{
\cleardoublepage
\pagestyle{empty}
\begin{titlepage}
	\makescucoverhead
	\begin{center}
		% 校名和论文类别、“四川大学”字样
		\makescucoverimage
		
		%封面论文标题
		\makescucovertitle

		%封面主要内容
		\makescucovercontent

		\end{center}
	\end{titlepage}
	\thispagestyle{empty}
}

%设置声明
\newcommand{\makestatement}{
\ifthenelse{\equal{\approval}{0}}{
	\cleardoublepage
	\pagestyle{empty}
	\begin{ENGinner}
	\end{ENGinner}
	
	% 中文声明
	{
		\cleardoublepage
		\setlength{\baselineskip}{20pt}
		\chapter*{声\hspace{1em}明}
		\thispagestyle{empty}
		本人声明所呈交的学位论文是本人在导师指导下（或联合培养导师组合作指导下）进行的研究工作及取得的研究成果。据我所知，除了文中特别加以标注和致谢的地方外，论文中不包含其他人已经发表或撰写过的研究成果，也不包含为获得四川大学或其他教育机构的学位或证书而使用过的材料。与我一同工作的同志对本研究所做的任何贡献均已在论文中作了明确的说明并表示谢意。

		本学位论文成果是本人在四川大学读书期间在导师指导下（或联合培养导师组合作指导下）取得的，论文成果归四川大学所有（或联合培养单位共有），特此声明。
		\vspace{4cm}
		\autograph
	}
}{}
% \ \\
}

\newcommand{\makethanks}{
\ifthenelse{\equal{\approval}{0}}{
	\include{Chapters/Thanks}
}{}
% \ \\
}


% 重定义autoref标签
%\renewcommand{\partautorefname}{\part}
%\renewcommand{\figureautorefname}{\figurename}
%\renewcommand{\tableautorefname}{\tablename}
%\renewcommand{\appendixautorefname}{\appendixname}
%\renewcommand{\equationautorefname}{\equationname}
%\renewcommand{\Itemautorefname}{\Itemname}

%\newcommand{\equationname}{式}
%\renewcommand{\equationautorefname}{\equationname}

% 定义\specialchap命令，生成没有编号的章节。从pkuthss-1.2beta保留，本模版中未使用。
\newcommand{\specialchap}[1]{
	\chapter*{#1}\markboth{#1}{}
	\addcontentsline{toc}{chapter}{#1}\phantomsection
}

% 定义\autograph命令，生成版权授权书等的签名栏。
\newcommand{\autograph}{
	\begin{center}
		\begin{tabular*}{\textwidth}{p{0.5\textwidth}p{0.5\textwidth}}
			\hspace{1.6em}学位论文作者签名：& 导师签名：\\[1cm]
			\hspace{1.6em}签字日期：\qquad 年 \quad 月 \quad 日 & 签字日期：\qquad 年 \quad 月 \quad 日
		\end{tabular*}
	\end{center}
}

\newenvironment{ENGinner}
{

	\renewcommand{\baselinestretch}{1.5}

	\begin{center}
		\ \\[12pt]
		\zihao{3}\rm
		{\parbox[c]{15.8cm}{\centering\textbf{\@ENGtitle}}}
		\\[4cm]
		\textbf{A dissertation submitted to Sichuan University} \\
		\textbf{in partial fulfillment of the requirements} \\
		\textbf{for the degree of } \\[14pt]
		\textbf{\@ENGdegreeclass} \\
		\ifthenelse{\equal{\@hasmajor}{1}}{
			\textbf{in \@ENGmajor}}
	\ \\[29pt]
	\textbf{By} \\
	\textbf{\@ENGauthor}\\
	\textbf{Supervisor: \@ENGsupervisor}\\[3.4cm]
	{\parbox[c]{15.8cm}{\centering\textbf{\@ENGschool, Sichuan University, Chengdu, China}}}\\[0.8cm]
	\textbf{\@statementdate}
	\end{center}
	\ \vspace{0cm}\par
}{}


% 定义中文摘要环境
\newenvironment{CHSabstract}
{
	\makefancyhf{摘要}
	\chapter{摘\hspace{1em}要}
	\zihao{-4}
	%\ \\[-0.5cm]
}
{
	\vspace{1em}\par
	\textbf{关键词：}\@keywords
	\cleardoublepage
	% \putonbottom{\hspace{1.8em}\textbf{关键词：}\@keywords}
}
% 定义英文摘要环境
\newenvironment{ENGabstract}
{
	\makefancyhf{Abstract}
	\rm
	\chapter{\textbf{Abstract}}
	\zihao{-4}
	% \ \\[-0.5cm]\setlength\parindent{2em}
}
{
	\vspace{1em}\par
	\textbf{Keywords:} \@ENGkeywords
	\cleardoublepage
	% \putonbottom{\hspace{1.8em}\textbf{Keywords:} \@ENGkeywords}
}
% 定义参考文献环境
\newenvironment{reference}
{
	\bibliographystyle{gbt7714-numerical}
	\cleardoublepage
	\zihao{5}
}

% 定义附录环境
\newenvironment{appendix*}
{
	\begin{center}
	\zihao{5}
}{\end{center}}

% 定义致谢环境
\newenvironment{thanks*}
{	
	\makefancyhf{致谢}
	\chapter{致\hspace{1em}谢}
}{}

% 设置文档默认行距为20pt
\setlength{\baselineskip}{20pt}
% 加载cleveref宏，支持智能化引用
\RequirePackage{cleveref}
\crefformat{chapter}{第~#2#1#3~章}
\Crefformat{chapter}{第~#2#1#3~章}
\crefrangeformat{chapter}{第~#3#1#4~-~#5#2#6~章}
\Crefrangeformat{chapter}{第~#3#1#4~-~#5#2#6~章}
\crefmultiformat{chapter}{第~#2#1#3~章}{和第~#2#1#3~章}{,~第~#2#1#3~章}{和第~#2#1#3~章}
\Crefmultiformat{chapter}{第~#2#1#3~章}{和第~#2#1#3~章}{,~第~#2#1#3~章}{和第~#2#1#3~章}
\crefrangemultiformat{chapter}{第~#3#1#4~-~#5#2#6~章}{和第~#3#1#4~-~#5#2#6~章}{,~ 第~#3#1#4~-~#5#2#6~章}{和第~#3#1#4~-~#5#2#6~章}
\Crefrangemultiformat{chapter}{第~#3#1#4~-~#5#2#6~章}{和第~#3#1#4~-~#5#2#6~章}{,~ 第~#3#1#4~-~#5#2#6~章}{和第~#3#1#4~-~#5#2#6~章}

\crefformat{section}{第~#2#1#3~节}
\Crefformat{section}{第~#2#1#3~节}
\crefrangeformat{section}{第~#3#1#4~-~#5#2#6~节}
\Crefrangeformat{section}{第~#3#1#4~-~#5#2#6~节}
\crefmultiformat{section}{第~#2#1#3~节}{和第~#2#1#3~节}{,~第~#2#1#3~节}{和第~#2#1#3~节}
\Crefmultiformat{section}{第~#2#1#3~节}{和第~#2#1#3~节}{,~第~#2#1#3~节}{和第~#2#1#3~节}
\crefrangemultiformat{section}{第~#3#1#4~-~#5#2#6~节}{和第~#3#1#4~-~#5#2#6~节}{,~ 第~#3#1#4~-~#5#2#6~节}{和第~#3#1#4~-~#5#2#6~节}
\Crefrangemultiformat{section}{第~#3#1#4~-~#5#2#6~节}{和第~#3#1#4~-~#5#2#6~节}{,~ 第~#3#1#4~-~#5#2#6~节}{和第~#3#1#4~-~#5#2#6~节}

\crefformat{subsection}{第~#2#1#3~小节}
\Crefformat{subsection}{第~#2#1#3~小节}
\crefrangeformat{subsection}{第~#3#1#4~-~#5#2#6~小节}
\Crefrangeformat{subsection}{第~#3#1#4~-~#5#2#6~小节}
\crefmultiformat{subsection}{第~#2#1#3~小节}{和第~#2#1#3~小节}{,~第~#2#1#3~小节}{和第~#2#1#3~小节}
\Crefmultiformat{subsection}{第~#2#1#3~小节}{和第~#2#1#3~小节}{,~第~#2#1#3~小节}{和第~#2#1#3~小节}
\crefrangemultiformat{subsection}{第~#3#1#4~-~#5#2#6~小节}{和第~#3#1#4~-~#5#2#6~小节}{,~ 第~#3#1#4~-~#5#2#6~小节}{和第~#3#1#4~-~#5#2#6~小节}
\Crefrangemultiformat{subsection}{第~#3#1#4~-~#5#2#6~小节}{和第~#3#1#4~-~#5#2#6~小节}{,~ 第~#3#1#4~-~#5#2#6~小节}{和第~#3#1#4~-~#5#2#6~小节}

\crefformat{subsubsection}{第~#2#1#3~项}
\Crefformat{subsubsection}{第~#2#1#3~项}
\crefrangeformat{subsubsection}{第~#3#1#4~-~#5#2#6~项}
\Crefrangeformat{subsubsection}{第~#3#1#4~-~#5#2#6~项}
\crefmultiformat{subsubsection}{第~#2#1#3~项}{和第~#2#1#3~项}{,~第~#2#1#3~项}{和第~#2#1#3~项}
\Crefmultiformat{subsubsection}{第~#2#1#3~项}{和第~#2#1#3~项}{,~第~#2#1#3~项}{和第~#2#1#3~项}
\crefrangemultiformat{subsubsection}{第~#3#1#4~-~#5#2#6~项}{和第~#3#1#4~-~#5#2#6~项}{,~ 第~#3#1#4~-~#5#2#6~项}{和第~#3#1#4~-~#5#2#6~项}
\Crefrangemultiformat{subsubsection}{第~#3#1#4~-~#5#2#6~项}{和第~#3#1#4~-~#5#2#6~项}{,~ 第~#3#1#4~-~#5#2#6~项}{和第~#3#1#4~-~#5#2#6~项}

\crefname{table}{表}{表}
\Crefname{table}{表}{表}
\creflabelformat{table}{#2#1#3~}
\crefrangelabelformat{table}{#3#1#4~-~#5#2#6~}
\crefmultiformat{table}{表~#2#1#3}{和表~#2#1#3}{,~表~#2#1#3}{和表~#2#1#3}
\Crefmultiformat{table}{表~#2#1#3}{和表~#2#1#3}{,~表~#2#1#3}{和表~#2#1#3}
\crefrangemultiformat{table}{表~#3#1#4~-~#5#2#6}{和表~#3#1#4~-~#5#2#6}{,~表~#3#1#4~-~#5#2#6}{和表~#3#1#4~-~#5#2#6}
\Crefrangemultiformat{table}{表~#3#1#4~-~#5#2#6}{和表~#3#1#4~-~#5#2#6}{,~表~#3#1#4~-~#5#2#6}{和表~#3#1#4~-~#5#2#6}

\crefname{figure}{图}{图}
\Crefname{figure}{图}{图}
\creflabelformat{figure}{#2#1#3~}
\crefrangelabelformat{figure}{#3#1#4~-~#5#2#6~}
\crefmultiformat{figure}{图~#2#1#3}{和图~#2#1#3}{,~图~#2#1#3}{和图~#2#1#3}
\Crefmultiformat{figure}{图~#2#1#3}{和图~#2#1#3}{,~图~#2#1#3}{和图~#2#1#3}
\crefrangemultiformat{figure}{图~#3#1#4~-~#5#2#6}{和图~#3#1#4~-~#5#2#6}{,~图~#3#1#4~-~#5#2#6}{和图~#3#1#4~-~#5#2#6}
\Crefrangemultiformat{figure}{图~#3#1#4~-~#5#2#6}{和图~#3#1#4~-~#5#2#6}{,~图~#3#1#4~-~#5#2#6}{和图~#3#1#4~-~#5#2#6}

\crefname{algorithm}{算法}{算法}
\Crefname{algorithm}{算法}{算法}
\creflabelformat{algorithm}{#2#1#3~}
\crefrangelabelformat{algorithm}{#3#1#4~-~#5#2#6~}
\crefmultiformat{algorithm}{算法~#2#1#3}{和算法~#2#1#3}{,~算法~#2#1#3}{和算法~#2#1#3}
\Crefmultiformat{algorithm}{算法~#2#1#3}{和算法~#2#1#3}{,~算法~#2#1#3}{和算法~#2#1#3}
\crefrangemultiformat{algorithm}{算法~#3#1#4~-~#5#2#6}{和算法~#3#1#4~-~#5#2#6}{,~算法~#3#1#4~-~#5#2#6}{和算法~#3#1#4~-~#5#2#6}
\Crefrangemultiformat{algorithm}{算法~#3#1#4~-~#5#2#6}{和算法~#3#1#4~-~#5#2#6}{,~算法~#3#1#4~-~#5#2#6}{和算法~#3#1#4~-~#5#2#6}

\crefname{equation}{式}{式}
\Crefname{equation}{式}{式}
\creflabelformat{equation}{(#2#1#3)~}
\crefrangelabelformat{equation}{(#3#1#4)~-~(#5#2#6)~}
\crefmultiformat{equation}{式~(#2#1#3)}{和式~(#2#1#3)}{,~式~(#2#1#3)}{和式~(#2#1#3)}
\Crefmultiformat{equation}{式~(#2#1#3)}{和式~(#2#1#3)}{,~式~(#2#1#3)}{和式~(#2#1#3)}
\crefrangemultiformat{equation}{式~(#3#1#4)~-~(#5#2#6)}{和式~(#3#1#4)~-~(#5#2#6)}{,~式~(#3#1#4)~-~(#5#2#6)}{和式~(#3#1#4)~-~(#5#2#6)}
\Crefrangemultiformat{equation}{式~(#3#1#4)~-~(#5#2#6)}{和式~(#3#1#4)~-~(#5#2#6)}{,~式~(#3#1#4)~-~(#5#2#6)}{和式~(#3#1#4)~-~(#5#2#6)}

\crefformat{page}{第~#2#1#3~页}
\Crefformat{page}{第~#2#1#3~页}
\crefrangeformat{page}{第~#3#1#4~-~#5#2#6~页}
\Crefrangeformat{page}{第~#3#1#4~-~#5#2#6~页}
\crefmultiformat{page}{第~#2#1#3~页}{和第~#2#1#3~页}{,~第~#2#1#3~页}{和第~#2#1#3~页}
\Crefmultiformat{page}{第~#2#1#3~页}{和第~#2#1#3~页}{,~第~#2#1#3~页}{和第~#2#1#3~页}
\crefrangemultiformat{page}{第~#3#1#4~-~#5#2#6~页}{和第~#3#1#4~-~#5#2#6~页}{,~ 第~#3#1#4~-~#5#2#6~页}{和第~#3#1#4~-~#5#2#6~页}
\Crefrangemultiformat{page}{第~#3#1#4~-~#5#2#6~页}{和第~#3#1#4~-~#5#2#6~页}{,~ 第~#3#1#4~-~#5#2#6~页}{和第~#3#1#4~-~#5#2#6~页}



\newcommand{\crefpairconjunction}{,}
\newcommand{\crefmiddleconjunction}{,}
\newcommand{\creflastconjunction}{,}
\newcommand{\crefrangeconjunction}{-}
% 加载配置文件
\AtEndOfClass{\input{./Template/scuthesis.def}}
\endinput
